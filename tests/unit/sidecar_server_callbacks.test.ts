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
          result_value = 0.5;
        } else if (p.method === '__call__' && p.handle === 'lm-1') {
          result_value = '```\ninitial prompt\n```';
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
});
