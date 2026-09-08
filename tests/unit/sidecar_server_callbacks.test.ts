import * as net from 'node:net';
import * as os from 'node:os';
import * as path from 'node:path';
import * as readline from 'node:readline';
import { describe, expect, it } from 'vitest';
import { create_server } from '../../src/sidecar/server.js';

const TEST_TIMEOUT = 15_000;

interface CapturedInvoke {
  handle: string;
  method: string;
  args: unknown[];
}

function make_socket_path(): string {
  return path.join(os.tmpdir(), `gepa-test-${Date.now()}-${Math.random().toString(36).slice(2)}.sock`);
}

async function run_via_server(params: unknown): Promise<{
  invokes: CapturedInvoke[];
  result: Record<string, unknown>;
}> {
  const socket_path = make_socket_path();
  const server = create_server();
  await new Promise<void>((resolve) => server.listen(socket_path, resolve));

  const invokes: CapturedInvoke[] = [];
  let train_fetch_count = 0;
  let val_fetch_count = 0;

  const result = await new Promise<Record<string, unknown>>((resolve, reject) => {
    const socket = net.createConnection(socket_path);
    const rl = readline.createInterface({ input: socket, crlfDelay: Infinity });

    socket.on('error', reject);

    rl.on('line', (line) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(line) as Record<string, unknown>;
      } catch {
        return;
      }

      if (msg['method'] === 'callback_invoke') {
        const p = msg['params'] as { handle: string; method: string; args: unknown[] };
        invokes.push({ handle: p.handle, method: p.method, args: p.args });

        // Evaluator __call__ → score; LM __call__ → a valid candidate string; callback methods → null
        let result_value: unknown = null;
        if (p.method === '__call__' && p.handle === 'eval-1') {
          const candidate = p.args[0];
          if (typeof candidate === 'object' && candidate !== null && 'number' in candidate) {
            const candidate_record = candidate as Record<string, unknown>;
            result_value = candidate_record['number'] === '42' ? [1, { scores: { accuracy: 1 } }] : [0, { scores: { accuracy: 0 } }];
          } else {
            result_value = 0.5;
          }
        } else if (p.method === '__call__' && p.handle === 'lm-1') {
          result_value = '```\nimproved prompt\n```';
        } else if (p.method === '__call__' && p.handle === 'refiner-lm-1') {
          result_value = '```json\n{"number":"42"}\n```';
        } else if (p.handle === 'adapter-1' && p.method === 'evaluate') {
          const batch = Array.isArray(p.args[0]) ? p.args[0] : [];
          const candidate = p.args[1];
          const capture_traces = p.args[2] === true;
          const prompt = typeof candidate === 'object' && candidate !== null
            ? (candidate as Record<string, unknown>)['system_prompt']
            : null;
          const score = prompt === 'improved prompt' ? 1 : 0;
          result_value = {
            outputs: batch.map(() => ({ answer: score === 1 ? 'ok' : 'bad' })),
            scores: batch.map(() => score),
            ...(capture_traces
              ? { trajectories: batch.map((item) => ({ item, feedback: score === 1 ? 'correct' : 'needs improvement' })) }
              : {}),
            objective_scores: batch.map(() => ({ exact: score })),
            num_metric_calls: batch.length,
          };
        } else if (p.handle === 'adapter-1' && p.method === 'make_reflective_dataset') {
          result_value = {
            system_prompt: [
              {
                Inputs: 'question',
                'Generated Outputs': 'bad',
                Feedback: 'needs improvement',
              },
            ],
          };
        } else if (p.handle === 'adapter-1' && p.method === 'propose_new_texts') {
          result_value = { system_prompt: 'improved prompt' };
        } else if (p.handle === 'val-policy-1' && p.method === 'get_eval_batch') {
          const state = p.args[1] as Record<string, unknown>;
          const evaluated = state['valset_evaluations'] as Record<string, unknown> | undefined;
          result_value = evaluated !== undefined && Object.prototype.hasOwnProperty.call(evaluated, '0') ? [1] : [0];
        } else if (p.handle === 'val-policy-1' && p.method === 'get_best_program') {
          result_value = 0;
        } else if (p.handle === 'val-policy-1' && p.method === 'get_valset_score') {
          const state = p.args[1] as Record<string, unknown>;
          const program_idx = typeof p.args[0] === 'number' ? p.args[0] : 0;
          const val_scores = state['prog_candidate_val_subscores'] as Array<Record<string, number>> | undefined;
          const scores = val_scores?.[program_idx] ?? {};
          const values = Object.values(scores);
          result_value = values.length === 0 ? Number.NEGATIVE_INFINITY : values.reduce((acc, score) => acc + score, 0) / values.length;
        } else if ((p.handle === 'train-loader-1' || p.handle === 'val-loader-1') && p.method === 'all_ids') {
          const fetch_count = p.handle === 'train-loader-1' ? train_fetch_count : val_fetch_count;
          result_value = fetch_count > 0 ? [0, 1, 2] : [0, 1];
        } else if ((p.handle === 'train-loader-1' || p.handle === 'val-loader-1') && p.method === 'fetch') {
          const ids = Array.isArray(p.args[0]) ? p.args[0] : [];
          if (p.handle === 'train-loader-1') {
            train_fetch_count += 1;
          } else {
            val_fetch_count += 1;
          }
          result_value = ids.map((id) => ({ id, source: p.handle }));
        }
        socket.write(JSON.stringify({ jsonrpc: '2.0', id: msg['id'], result: result_value }) + '\n');
        return;
      }

      // Final result or error from the server
      if (msg['id'] !== undefined && msg['method'] === undefined) {
        if (msg['error'] !== undefined) {
          reject(new Error((msg['error'] as { message: string }).message));
        } else {
          resolve(msg['result'] as Record<string, unknown>);
        }
        socket.destroy();
      }
    });

    socket.write(
      JSON.stringify({ jsonrpc: '2.0', id: 'rpc-1', method: 'optimize_anything', params }) + '\n',
    );
  });

  await new Promise<void>((resolve) => server.close(() => resolve()));
  return { invokes, result };
}

describe('create_server — callback_handles protocol', () => {
  it('fires on_optimization_start before final result when callback_handles provided', async () => {
    const { invokes, result } = await run_via_server({
      seed_candidate: 'initial prompt',
      dataset: [{ input: 'a' }, { input: 'b' }],
      valset: [{ input: 'c' }],
      objective: null,
      background: null,
      config: { engine: { max_metric_calls: 6 } },
      evaluator_handle: 'eval-1',
      reflection_lm_handle: 'lm-1',
      callback_handles: [
        { id: 'cb-1', methods: ['on_optimization_start', 'on_optimization_end'] },
      ],
    });

    const start_invoke = invokes.find(
      (inv) => inv.handle === 'cb-1' && inv.method === 'on_optimization_start',
    );
    expect(start_invoke).toBeDefined();
    expect(Array.isArray(start_invoke?.args)).toBe(true);
    expect((start_invoke?.args as unknown[]).length).toBeGreaterThanOrEqual(1);

    // val_subscores must be serialized as an array of plain objects, not Maps
    const val_subscores = result['val_subscores'];
    expect(Array.isArray(val_subscores)).toBe(true);
    if ((val_subscores as unknown[]).length > 0) {
      const first = (val_subscores as unknown[])[0];
      expect(first instanceof Map).toBe(false);
      expect(typeof first).toBe('object');
    }

    // per_val_instance_best_candidates must be a plain object
    const per_val = result['per_val_instance_best_candidates'];
    expect(per_val instanceof Map).toBe(false);
    expect(typeof per_val).toBe('object');

    // parents must be present
    expect(Array.isArray(result['parents'])).toBe(true);
    expect(result['validation_schema_version']).toBe(2);
    expect(result['best_idx']).toBeTypeOf('number');
  }, TEST_TIMEOUT);

  it('evaluator __call__ uses method field on callback_invoke', async () => {
    const { invokes } = await run_via_server({
      seed_candidate: 'x',
      dataset: null,
      valset: null,
      objective: null,
      background: null,
      config: { engine: { max_metric_calls: 3 } },
      evaluator_handle: 'eval-1',
      reflection_lm_handle: 'lm-1',
    });

    const eval_invokes = invokes.filter((inv) => inv.handle === 'eval-1');
    expect(eval_invokes.length).toBeGreaterThan(0);
    for (const inv of eval_invokes) {
      expect(inv.method).toBe('__call__');
    }
  }, TEST_TIMEOUT);

  it('routes refiner_lm_handle through callback_invoke separately from reflection_lm_handle', async () => {
    const { invokes, result } = await run_via_server({
      seed_candidate: { number: '10', refiner_prompt: 'Improve the number.' },
      dataset: null,
      valset: null,
      objective: null,
      background: null,
      config: {
        engine: { max_metric_calls: 1 },
        refiner: { refiner_lm_handle: 'refiner-lm-1', max_refinements: 1 },
      },
      evaluator_handle: 'eval-1',
      reflection_lm_handle: 'lm-1',
    });

    expect(invokes.some((inv) => inv.handle === 'refiner-lm-1' && inv.method === '__call__')).toBe(true);
    expect(result['total_metric_calls']).toBe(2);
    expect(result['val_aggregate_scores']).toEqual([1]);
  }, TEST_TIMEOUT);

  it('uses adapter_handle as the optimization adapter without an evaluator_handle', async () => {
    const { invokes, result } = await run_via_server({
      seed_candidate: { system_prompt: 'seed prompt' },
      dataset: [{ input: 'train-1' }, { input: 'train-2' }],
      valset: [{ input: 'val-1' }, { input: 'val-2' }],
      objective: null,
      background: null,
      config: {
        engine: { max_metric_calls: 8 },
        reflection: { reflection_minibatch_size: 2 },
      },
      adapter_handle: 'adapter-1',
      reflection_lm_handle: 'lm-1',
    });

    expect(invokes.some((inv) => inv.handle === 'adapter-1' && inv.method === 'evaluate')).toBe(true);
    expect(invokes.some((inv) => inv.handle === 'adapter-1' && inv.method === 'make_reflective_dataset')).toBe(true);
    expect(invokes.some((inv) => inv.handle === 'eval-1')).toBe(false);
    expect(result['best_candidate']).toEqual({ system_prompt: 'improved prompt' });
    expect(result['val_aggregate_scores']).toEqual([0, 1]);
  }, TEST_TIMEOUT);

  it('uses remote adapter propose_new_texts without calling the reflection LM', async () => {
    const { invokes, result } = await run_via_server({
      seed_candidate: { system_prompt: 'seed prompt' },
      dataset: [{ input: 'train-1' }, { input: 'train-2' }],
      valset: [{ input: 'val-1' }, { input: 'val-2' }],
      objective: null,
      background: null,
      config: {
        engine: { max_metric_calls: 8 },
        reflection: { reflection_minibatch_size: 2 },
      },
      adapter_handle: 'adapter-1',
      adapter_propose_new_texts: true,
      reflection_lm_handle: 'lm-1',
    });

    expect(invokes.some((inv) => inv.handle === 'adapter-1' && inv.method === 'propose_new_texts')).toBe(true);
    expect(invokes.some((inv) => inv.handle === 'lm-1')).toBe(false);
    expect(result['best_candidate']).toEqual({ system_prompt: 'improved prompt' });
  }, TEST_TIMEOUT);

  it('uses remote loader handles with refreshed ids after fetch side effects', async () => {
    const { invokes, result } = await run_via_server({
      seed_candidate: { system_prompt: 'seed prompt' },
      dataset: null,
      valset: null,
      dataset_loader_handle: 'train-loader-1',
      valset_loader_handle: 'val-loader-1',
      objective: null,
      background: null,
      config: {
        engine: { max_metric_calls: 10 },
        reflection: { reflection_minibatch_size: 2 },
      },
      adapter_handle: 'adapter-1',
      reflection_lm_handle: 'lm-1',
    });

    expect(invokes.some((inv) => inv.handle === 'train-loader-1' && inv.method === 'all_ids')).toBe(true);
    expect(invokes.some((inv) => inv.handle === 'train-loader-1' && inv.method === 'fetch')).toBe(true);
    expect(invokes.some((inv) => inv.handle === 'val-loader-1' && inv.method === 'all_ids')).toBe(true);
    expect(invokes.some((inv) => inv.handle === 'val-loader-1' && inv.method === 'fetch')).toBe(true);
    expect(result['best_candidate']).toEqual({ system_prompt: 'improved prompt' });

    const val_subscores = result['val_subscores'];
    expect(Array.isArray(val_subscores)).toBe(true);
    const accepted_scores = (val_subscores as Array<Record<string, number>>)[1];
    expect(accepted_scores).toEqual({ 0: 1, 1: 1, 2: 1 });
  }, TEST_TIMEOUT);

  it('uses remote validation policy handles for sparse valset evaluation', async () => {
    const { invokes, result } = await run_via_server({
      seed_candidate: { system_prompt: 'seed prompt' },
      dataset: [{ input: 'train-1' }, { input: 'train-2' }],
      valset: [{ input: 'val-1' }, { input: 'val-2' }],
      objective: null,
      background: null,
      config: {
        engine: { max_metric_calls: 8 },
        reflection: { reflection_minibatch_size: 2 },
      },
      adapter_handle: 'adapter-1',
      adapter_propose_new_texts: true,
      val_evaluation_policy_handle: 'val-policy-1',
      reflection_lm_handle: 'lm-1',
    });

    expect(invokes.some((inv) => inv.handle === 'val-policy-1' && inv.method === 'get_eval_batch')).toBe(true);
    expect(invokes.some((inv) => inv.handle === 'val-policy-1' && inv.method === 'get_best_program')).toBe(true);
    expect(invokes.some((inv) => inv.handle === 'val-policy-1' && inv.method === 'get_valset_score')).toBe(true);
    expect(result['val_subscores']).toEqual([{ 0: 0, 1: 0 }, { 1: 1 }, { 1: 1 }]);
  }, TEST_TIMEOUT);
});
