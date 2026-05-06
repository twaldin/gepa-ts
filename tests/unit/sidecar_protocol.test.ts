import { describe, expect, it } from 'vitest';
import { is_optimize_request } from '../../src/sidecar/protocol.js';
import type { EvaluatorCtx } from '../../src/sidecar/protocol.js';

const BASE_PARAMS = {
  seed_candidate: 'test',
  dataset: null,
  valset: null,
  objective: null,
  background: null,
  config: {},
  evaluator_handle: 'eval-1',
  reflection_lm_handle: null,
};

const BASE_REQ = {
  jsonrpc: '2.0' as const,
  id: 1,
  method: 'optimize_anything',
  params: BASE_PARAMS,
};

describe('is_optimize_request', () => {
  it('accepts a valid request without callback_handles', () => {
    expect(is_optimize_request(BASE_REQ)).toBe(true);
  });

  it('accepts a valid request with non-empty callback_handles', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        callback_handles: [
          { id: 'cb-1', methods: ['on_optimization_start', 'on_optimization_end'] },
        ],
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('accepts a valid request with empty callback_handles array', () => {
    const req = {
      ...BASE_REQ,
      params: { ...BASE_PARAMS, callback_handles: [] },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects callback_handles entry with non-string id', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        callback_handles: [{ id: 123, methods: ['on_optimization_start'] }],
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('rejects callback_handles entry where methods is not an array', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        callback_handles: [{ id: 'cb-1', methods: 'on_optimization_start' }],
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('rejects callback_handles entry with non-string method item', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        callback_handles: [{ id: 'cb-1', methods: [42] }],
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('rejects callback_handles that is not an array', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        callback_handles: { id: 'cb-1', methods: [] },
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });
});

describe('EvaluatorCtx', () => {
  it('accepts ctx without opt_state', () => {
    const ctx: EvaluatorCtx = { example: { input: 'hello' } };
    expect(ctx.example).toEqual({ input: 'hello' });
    expect(ctx.opt_state).toBeUndefined();
  });

  it('accepts ctx with opt_state', () => {
    const ctx: EvaluatorCtx = {
      example: { input: 'hello' },
      opt_state: {
        best_example_evals: [{ score: 0.9, side_info: { key: 'value' } }],
      },
    };
    expect(ctx.opt_state?.best_example_evals[0]?.score).toBe(0.9);
  });

  it('allows opt_state with empty best_example_evals', () => {
    const ctx: EvaluatorCtx = {
      example: null,
      opt_state: { best_example_evals: [] },
    };
    expect(ctx.opt_state?.best_example_evals).toHaveLength(0);
  });
});
