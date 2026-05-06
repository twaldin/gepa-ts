import * as net from 'node:net';
import * as readline from 'node:readline';
import { optimize_anything } from '../index.js';
import { EvaluatorWrapper } from '../evaluator_wrapper.js';
import { LogContext, runWithLogContext } from '../log_context.js';
import type { Evaluator, LanguageModel, GEPAConfig, EvalResult, SideInfo, DataId, GEPACallback } from '../types.js';
import type { JsonRpcResponse, CallbackInvokeParams, EvaluatorCtx } from './protocol.js';
import { is_json_rpc_response, is_optimize_request } from './protocol.js';
import type { GEPAResult } from '../result.js';

let _cb_counter = 0;
function next_cb_id(): string { return `srv-${++_cb_counter}`; }

function to_side_info(val: unknown): SideInfo { return val !== null && typeof val === 'object' ? (val as Record<string, unknown>) : {}; }

function parse_eval_result(raw: unknown): EvalResult {
  if (typeof raw === 'number') return raw;
  if (Array.isArray(raw) && typeof raw[0] === 'number') {
    const side = raw.length >= 3 ? raw[2] : raw[1];
    return [raw[0], to_side_info(side)];
  }
  throw new Error(`Unexpected evaluator result: ${JSON.stringify(raw)}`);
}

function map_to_record<V>(m: Map<DataId, V>): Record<string, V> { const out: Record<string, V> = {}; for (const [k, v] of m) out[String(k)] = v; return out; }
function map_set_to_record(m: Map<DataId, Set<number>>): Record<string, number[]> { const out: Record<string, number[]> = {}; for (const [k, v] of m) out[String(k)] = Array.from(v); return out; }
function serialize_result(result: GEPAResult): Record<string, unknown> {
  return {
    candidates: result.candidates, parents: result.parents, val_aggregate_scores: result.val_aggregate_scores,
    val_subscores: result.val_subscores.map(map_to_record),
    per_val_instance_best_candidates: map_set_to_record(result.per_val_instance_best_candidates),
    per_objective_best_candidates: result.per_objective_best_candidates !== null ? map_set_to_record(result.per_objective_best_candidates as Map<DataId, Set<number>>) : null,
    best_outputs_valset: result.best_outputs_valset !== null ? map_to_record(result.best_outputs_valset) : null,
    discovery_eval_counts: result.discovery_eval_counts, val_aggregate_subscores: result.val_aggregate_subscores,
    objective_pareto_front: result.objective_pareto_front, num_full_val_evals: result.num_full_val_evals,
    run_dir: result.run_dir, _str_candidate_key: result._str_candidate_key, best_candidate: result.best_candidate,
    best_idx: result.best_idx, total_metric_calls: result.total_metric_calls, num_candidates: result.num_candidates,
    num_val_instances: result.num_val_instances, seed: result.seed,
  };
}

function build_remote_callback(handle: string, methods: string[], send: (h: string, m: string, a: unknown[]) => void): GEPACallback {
  const cb: Partial<Required<GEPACallback>> = {};
  for (const method of methods) {
    switch (method) {
      case 'on_optimization_start': cb.on_optimization_start = (e) => { send(handle, method, [e]); }; break;
      case 'on_optimization_end': cb.on_optimization_end = (e) => { send(handle, method, [e]); }; break;
      case 'on_iteration_start': cb.on_iteration_start = (e) => { send(handle, method, [e]); }; break;
      case 'on_iteration_end': cb.on_iteration_end = (e) => { send(handle, method, [e]); }; break;
      case 'on_candidate_selected': cb.on_candidate_selected = (e) => { send(handle, method, [e]); }; break;
      case 'on_minibatch_sampled': cb.on_minibatch_sampled = (e) => { send(handle, method, [e]); }; break;
      case 'on_evaluation_start': cb.on_evaluation_start = (e) => { send(handle, method, [e]); }; break;
      case 'on_evaluation_end': cb.on_evaluation_end = (e) => { send(handle, method, [e]); }; break;
      case 'on_evaluation_skipped': cb.on_evaluation_skipped = (e) => { send(handle, method, [e]); }; break;
      case 'on_valset_evaluated': cb.on_valset_evaluated = (e) => { send(handle, method, [e]); }; break;
      case 'on_reflective_dataset_built': cb.on_reflective_dataset_built = (e) => { send(handle, method, [e]); }; break;
      case 'on_proposal_start': cb.on_proposal_start = (e) => { send(handle, method, [e]); }; break;
      case 'on_proposal_end': cb.on_proposal_end = (e) => { send(handle, method, [e]); }; break;
      case 'on_candidate_accepted': cb.on_candidate_accepted = (e) => { send(handle, method, [e]); }; break;
      case 'on_candidate_rejected': cb.on_candidate_rejected = (e) => { send(handle, method, [e]); }; break;
      case 'on_merge_attempted': cb.on_merge_attempted = (e) => { send(handle, method, [e]); }; break;
      case 'on_merge_accepted': cb.on_merge_accepted = (e) => { send(handle, method, [e]); }; break;
      case 'on_merge_rejected': cb.on_merge_rejected = (e) => { send(handle, method, [e]); }; break;
      case 'on_pareto_front_updated': cb.on_pareto_front_updated = (e) => { send(handle, method, [e]); }; break;
      case 'on_state_saved': cb.on_state_saved = (e) => { send(handle, method, [e]); }; break;
      case 'on_budget_updated': cb.on_budget_updated = (e) => { send(handle, method, [e]); }; break;
      case 'on_error': cb.on_error = (e) => { send(handle, method, [e]); }; break;
    }
  }
  return cb;
}

export function create_server(): net.Server {
  return net.createServer((socket: net.Socket) => {
    const pending = new Map<string | number, (r: JsonRpcResponse) => void>();
    const logContexts = new Map<string, LogContext>();
    const threadContexts = new Map<number, LogContext | null>();
    const wrappers = new Map<string, EvaluatorWrapper>();
    const rl = readline.createInterface({ input: socket, crlfDelay: Infinity });
    let handleCounter = 0;

    function nextHandle(prefix: string): string { handleCounter += 1; return `${prefix}-${handleCounter}`; }
    function write_line(obj: unknown): void { socket.write(JSON.stringify(obj) + '\n'); }

    function send_remote_call(handle: string, method: string, args: unknown[]): Promise<unknown> {
      const id = next_cb_id();
      return new Promise<unknown>((resolve, reject) => {
        pending.set(id, (r: JsonRpcResponse) => r.error !== undefined ? reject(new Error(r.error.message)) : resolve(r.result));
        write_line({ jsonrpc: '2.0', id, method: 'callback_invoke', params: { handle, method, args } as CallbackInvokeParams });
      });
    }

    function getThreadCtx(params: Record<string, unknown>): LogContext | null {
      const tid = params['client_thread_id'];
      if (typeof tid !== 'number') return null;
      return threadContexts.get(tid) ?? null;
    }

    rl.on('line', (line: string) => {
      let parsed: unknown;
      try { parsed = JSON.parse(line); } catch { return; }

      if (is_json_rpc_response(parsed)) {
        const resolver = pending.get(parsed.id);
        if (resolver !== undefined) { pending.delete(parsed.id); resolver(parsed); }
        return;
      }

      const req = parsed as Record<string, unknown>;
      const id = req['id'] ?? 0;
      const method = req['method'];
      const params = (req['params'] ?? {}) as Record<string, unknown>;
      if (req['jsonrpc'] !== '2.0' || (typeof method !== 'string')) {
        write_line({ jsonrpc: '2.0', id, error: { code: -32600, message: 'Invalid Request' } });
        return;
      }

      runWithLogContext(getThreadCtx(params), () => {
        if (method === 'optimize_anything') {
          if (!is_optimize_request(parsed)) {
            write_line({ jsonrpc: '2.0', id, error: { code: -32600, message: 'Invalid Request' } });
            return;
          }
          const p = parsed.params;
          const evaluator = ((candidate: string | Record<string, string>, ctx?: EvaluatorCtx): Promise<EvalResult> => {
            const args: unknown[] = ctx !== undefined ? [candidate, { example: ctx.example, opt_state: ctx.opt_state }] : [candidate];
            return send_remote_call(p.evaluator_handle, '__call__', args).then(parse_eval_result);
          }) as Evaluator;

          const reflection_lm: LanguageModel | undefined = p.reflection_lm_handle !== null
            ? (prompt) => send_remote_call(p.reflection_lm_handle as string, '__call__', [prompt]).then((r) => {
                if (typeof r !== 'string') throw new Error(`LM must return string, got ${typeof r}`);
                return r;
              })
            : undefined;

          const raw_engine = p.config.engine ?? {};
          const raw_reflection = p.config.reflection ?? {};
          const engine_config: GEPAConfig['engine'] = {
            ...(typeof raw_engine.max_metric_calls === 'number' ? { max_metric_calls: raw_engine.max_metric_calls } : {}),
            ...(typeof raw_engine.seed === 'number' ? { seed: raw_engine.seed } : {}),
            ...(typeof raw_engine.best_example_evals_k === 'number' ? { best_example_evals_k: raw_engine.best_example_evals_k } : {}),
          };
          const reflection_config: GEPAConfig['reflection'] = {
            ...(reflection_lm !== undefined ? { reflection_lm } : {}),
            ...(typeof raw_reflection.reflection_minibatch_size === 'number' ? { reflection_minibatch_size: raw_reflection.reflection_minibatch_size } : {}),
            ...(raw_reflection.reflection_prompt_template !== undefined ? { reflection_prompt_template: raw_reflection.reflection_prompt_template as string | Record<string, string> } : {}),
          };

          const callback_handles = p.callback_handles ?? [];
          const callbacks: GEPACallback[] = callback_handles.map(({ id: h, methods }) =>
            build_remote_callback(h, methods, (handle, m, args) => { send_remote_call(handle, m, args).catch(() => undefined); }),
          );
          const gepa_config: GEPAConfig = { engine: engine_config, reflection: reflection_config, ...(callbacks.length > 0 ? { callbacks } : {}) };

          optimize_anything({
            seed_candidate: p.seed_candidate ?? '', evaluator, dataset: p.dataset, valset: p.valset,
            objective: p.objective, background: p.background, config: gepa_config,
          }).then((result) => {
            write_line({ jsonrpc: '2.0', id, result: serialize_result(result) });
          }).catch((err: unknown) => {
            const message = err instanceof Error ? err.message : String(err);
            write_line({ jsonrpc: '2.0', id, error: { code: -32000, message } });
          });
          return;
        }

        if (method === 'log_context.create') {
          const handle = nextHandle('logctx');
          logContexts.set(handle, new LogContext());
          write_line({ jsonrpc: '2.0', id, result: handle });
          return;
        }
        if (method === 'log_context.write') {
          const handle = params['handle'];
          const text = params['text'];
          if (typeof handle !== 'string' || typeof text !== 'string') {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params' } });
            return;
          }
          const ctx = logContexts.get(handle);
          if (ctx === undefined) {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params' } });
            return;
          }
          ctx.write(text);
          write_line({ jsonrpc: '2.0', id, result: null });
          return;
        }
        if (method === 'log_context.drain') {
          const handle = params['handle'];
          if (typeof handle !== 'string') {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params' } });
            return;
          }
          const ctx = logContexts.get(handle);
          if (ctx === undefined) {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params' } });
            return;
          }
          write_line({ jsonrpc: '2.0', id, result: ctx.drain() });
          return;
        }
        if (method === 'set_log_context') {
          const tid = params['client_thread_id'];
          const handle = params['handle'];
          if (typeof tid !== 'number') {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'client_thread_id required' } });
            return;
          }
          threadContexts.set(tid, handle === null ? null : (typeof handle === 'string' ? (logContexts.get(handle) ?? null) : null));
          write_line({ jsonrpc: '2.0', id, result: null });
          return;
        }
        if (method === 'evaluator_wrapper.create') {
          const evaluatorHandle = params['evaluator_handle'];
          const singleInstanceMode = params['single_instance_mode'];
          if (typeof evaluatorHandle !== 'string' || typeof singleInstanceMode !== 'boolean') {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params' } });
            return;
          }
          const evaluator: Evaluator = ((candidate: string | Record<string, string>, ctx?: EvaluatorCtx) => {
            const args: unknown[] = ctx !== undefined ? [candidate, ctx] : [candidate];
            return send_remote_call(evaluatorHandle, '__call__', args).then(parse_eval_result);
          }) as Evaluator;
          const wrapper = new EvaluatorWrapper(
            evaluator,
            singleInstanceMode,
            params['capture_stdio'] === true,
            params['str_candidate_mode'] === true,
            params['raise_on_exception'] !== false,
          );
          const handle = nextHandle('evalwrap');
          wrappers.set(handle, wrapper);
          write_line({ jsonrpc: '2.0', id, result: handle });
          return;
        }
        if (method === 'evaluator_wrapper.call') {
          const handle = params['handle'];
          const candidate = params['candidate'];
          if (typeof handle !== 'string' || typeof candidate !== 'object' || candidate === null) {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params' } });
            return;
          }
          const wrapper = wrappers.get(handle);
          if (wrapper === undefined) {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params' } });
            return;
          }
          const opt = params['opt_state'];
          const best =
            typeof opt === 'object' && opt !== null && Array.isArray((opt as { best_example_evals?: unknown }).best_example_evals)
              ? ((opt as { best_example_evals: Array<{ score: number; side_info: Record<string, unknown> }> }).best_example_evals)
              : [];
          wrapper.call(candidate as Record<string, string>, params['example'], { best_example_evals: best }).then((ret) => {
            write_line({ jsonrpc: '2.0', id, result: ret });
          }).catch((err: unknown) => {
            const message = err instanceof Error ? err.message : String(err);
            write_line({ jsonrpc: '2.0', id, error: { code: -32000, message } });
          });
          return;
        }

        write_line({ jsonrpc: '2.0', id, error: { code: -32601, message: 'Method not found' } });
      });
    });

    socket.on('error', () => rl.close());
  });
}
