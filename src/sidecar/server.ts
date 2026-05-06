import * as net from 'node:net';
import * as readline from 'node:readline';
import { optimize_anything } from '../index.js';
import type { Evaluator, LanguageModel, GEPAConfig, EvalResult, SideInfo } from '../types.js';
import type { JsonRpcResponse, CallbackInvokeParams } from './protocol.js';
import { is_json_rpc_response, is_optimize_request } from './protocol.js';
import type { GEPAResult } from '../result.js';

let _cb_counter = 0;
function next_cb_id(): string {
  return `srv-${++_cb_counter}`;
}

function to_side_info(val: unknown): SideInfo {
  if (val !== null && typeof val === 'object') {
    return val as Record<string, unknown>;
  }
  return {};
}

function parse_eval_result(raw: unknown): EvalResult {
  if (typeof raw === 'number') return raw;
  if (Array.isArray(raw)) {
    const first = raw[0];
    const second = raw[1];
    if (typeof first === 'number') {
      return [first, to_side_info(second)];
    }
  }
  throw new Error(`Unexpected evaluator result: ${JSON.stringify(raw)}`);
}

function serialize_result(result: GEPAResult): Record<string, unknown> {
  return {
    candidates: result.candidates,
    val_aggregate_scores: result.val_aggregate_scores,
    best_candidate: result.best_candidate,
    best_idx: result.best_idx,
    total_metric_calls: result.total_metric_calls,
    num_candidates: result.num_candidates,
    num_val_instances: result.num_val_instances,
    seed: result.seed,
  };
}

export function create_server(): net.Server {
  return net.createServer((socket) => {
    const pending = new Map<string | number, (r: JsonRpcResponse) => void>();

    const rl = readline.createInterface({ input: socket, crlfDelay: Infinity });

    function write_line(obj: unknown): void {
      socket.write(JSON.stringify(obj) + '\n');
    }

    function send_callback(handle: string, args: unknown[]): Promise<unknown> {
      const id = next_cb_id();
      return new Promise<unknown>((resolve, reject) => {
        pending.set(id, (r: JsonRpcResponse) => {
          if (r.error !== undefined) {
            reject(new Error(r.error.message));
          } else {
            resolve(r.result);
          }
        });
        const msg: { jsonrpc: '2.0'; id: string; method: string; params: CallbackInvokeParams } = {
          jsonrpc: '2.0',
          id,
          method: 'callback_invoke',
          params: { handle, args },
        };
        write_line(msg);
      });
    }

    rl.on('line', (line) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(line);
      } catch {
        return;
      }

      // Response to a pending callback
      if (is_json_rpc_response(parsed)) {
        const resolver = pending.get(parsed.id);
        if (resolver !== undefined) {
          pending.delete(parsed.id);
          resolver(parsed);
          return;
        }
      }

      if (!is_optimize_request(parsed)) {
        write_line({ jsonrpc: '2.0', id: 0, error: { code: -32600, message: 'Invalid Request' } });
        return;
      }

      const { id, params } = parsed;

      const evaluator_handle = params.evaluator_handle;
      const evaluator = ((candidate: string | Record<string, string>, ctx?: { example: unknown }): Promise<EvalResult> => {
        const args: unknown[] = ctx !== undefined ? [candidate, ctx] : [candidate];
        return send_callback(evaluator_handle, args).then(parse_eval_result);
      }) as Evaluator;

      const lm_handle = params.reflection_lm_handle;
      const reflection_lm: LanguageModel | undefined = lm_handle !== null
        ? (prompt) => send_callback(lm_handle, [prompt]).then((r) => {
            if (typeof r !== 'string') throw new Error(`LM must return string, got ${typeof r}`);
            return r;
          })
        : undefined;

      const raw_engine = params.config.engine ?? {};
      const raw_reflection = params.config.reflection ?? {};

      const engine_config: GEPAConfig['engine'] = {
        ...(typeof raw_engine.max_metric_calls === 'number' ? { max_metric_calls: raw_engine.max_metric_calls } : {}),
        ...(typeof raw_engine.seed === 'number' ? { seed: raw_engine.seed } : {}),
      };

      const reflection_config: GEPAConfig['reflection'] = {
        ...(reflection_lm !== undefined ? { reflection_lm } : {}),
        ...(typeof raw_reflection.reflection_minibatch_size === 'number'
          ? { reflection_minibatch_size: raw_reflection.reflection_minibatch_size }
          : {}),
      };

      const gepa_config: GEPAConfig = {
        engine: engine_config,
        reflection: reflection_config,
      };

      const seed_candidate = params.seed_candidate ?? '';

      optimize_anything({
        seed_candidate,
        evaluator,
        dataset: params.dataset,
        valset: params.valset,
        objective: params.objective,
        background: params.background,
        config: gepa_config,
      }).then((result) => {
        write_line({ jsonrpc: '2.0', id, result: serialize_result(result) });
      }).catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err);
        write_line({ jsonrpc: '2.0', id, error: { code: -32000, message } });
      });
    });

    socket.on('error', () => {
      rl.close();
    });
  });
}
