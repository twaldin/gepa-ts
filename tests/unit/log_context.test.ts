import { describe, expect, test, vi } from 'vitest';
import { EvaluatorWrapper } from '../../src/evaluator_wrapper';
import { LogContext, getLogContext, getLogContextOrThrow, oa_log, runWithLogContext } from '../../src/log_context';

describe('log context + oa_log', () => {
  test('basic capture', () => {
    const ctx = new LogContext();
    runWithLogContext(ctx, () => oa_log('hello'));
    expect(ctx.drain()).toBe('hello\n');
  });

  test('custom sep/end', () => {
    const ctx = new LogContext();
    runWithLogContext(ctx, () => oa_log('a', 'b', { sep: '-', end: '!' }));
    expect(ctx.drain()).toBe('a-b!');
  });

  test('outside warns and discards', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    oa_log('x');
    oa_log('y');
    expect(spy).toHaveBeenCalledTimes(1);
    spy.mockRestore();
  });

  test('get outside throws', () => {
    expect(getLogContext()).toBeNull();
    expect(() => getLogContextOrThrow()).toThrow('No active log context');
  });

  test('wrapper injects log key only when non-empty', async () => {
    const w1 = new EvaluatorWrapper(() => 1.0, true);
    expect((await w1.call({ p: 'x' }))[2]).toEqual({});

    const w2 = new EvaluatorWrapper(() => {
      oa_log('hit');
      return 1.0;
    }, true);
    expect((await w2.call({ p: 'x' }))[2]).toEqual({ log: 'hit\n' });
  });

  test('async propagation', async () => {
    const wrapper = new EvaluatorWrapper(async () => {
      await Promise.resolve();
      oa_log('child');
      return 1;
    }, true);
    const out = await wrapper.call({ p: 'x' });
    expect(out[2]).toEqual({ log: 'child\n' });
  });

  test('parallel wrappers do not cross-contaminate', async () => {
    const wrapper = new EvaluatorWrapper(async (candidate) => {
      await new Promise((r) => setTimeout(r, candidate === 'a' ? 10 : 1));
      oa_log(candidate);
      return 1;
    }, true, false, true);
    const [a, b] = await Promise.all([wrapper.call({ current_candidate: 'a' }, undefined), wrapper.call({ current_candidate: 'b' }, undefined)]);
    const logs = [a[2].log, b[2].log].sort();
    expect(logs).toEqual(['a\n', 'b\n']);
  });

  test('concurrent writes safe', async () => {
    const wrapper = new EvaluatorWrapper(async () => {
      await Promise.all(Array.from({ length: 50 }, (_, i) => Promise.resolve().then(() => oa_log(i, { end: '' }))));
      return 1;
    }, true);
    const out = await wrapper.call({ p: 'x' });
    expect(typeof out[2].log).toBe('string');
    expect((out[2].log as string).length).toBeGreaterThan(10);
  });
});
