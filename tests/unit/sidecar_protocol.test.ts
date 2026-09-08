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

  it('accepts adapter_handle in place of evaluator_handle', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        evaluator_handle: undefined,
        adapter_handle: 'adapter-1',
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('accepts adapter_propose_new_texts when it is boolean', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        adapter_handle: 'adapter-1',
        adapter_propose_new_texts: true,
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects non-boolean adapter_propose_new_texts', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        adapter_handle: 'adapter-1',
        adapter_propose_new_texts: 'true',
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('accepts remote validation policy handles', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        val_evaluation_policy_handle: 'val-policy-1',
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects non-string remote validation policy handles', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        val_evaluation_policy_handle: 1,
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('accepts remote dataset and valset loader handles', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        dataset: null,
        valset: null,
        dataset_loader_handle: 'train-loader',
        valset_loader_handle: 'val-loader',
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects non-string remote loader handles', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        dataset_loader_handle: 1,
        valset_loader_handle: {},
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('rejects requests without evaluator_handle or adapter_handle', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        evaluator_handle: undefined,
        adapter_handle: undefined,
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('accepts numeric max_candidate_proposals in engine config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { engine: { max_candidate_proposals: 2 } },
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects non-numeric max_candidate_proposals in engine config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { engine: { max_candidate_proposals: '2' } },
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('accepts numeric max_reflection_cost in engine config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { engine: { max_reflection_cost: 0.25 } },
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects non-numeric max_reflection_cost in engine config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { engine: { max_reflection_cost: '0.25' } },
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('accepts boolean track_best_outputs in engine config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { engine: { track_best_outputs: true } },
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects non-boolean track_best_outputs in engine config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { engine: { track_best_outputs: 'true' } },
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('accepts boolean raise_on_exception and capture_stdio in engine config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { engine: { raise_on_exception: false, capture_stdio: true } },
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects non-boolean raise_on_exception and capture_stdio in engine config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { engine: { raise_on_exception: 'false', capture_stdio: 'true' } },
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('accepts string acceptance_criterion in engine config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { engine: { acceptance_criterion: 'improvement_or_equal' } },
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects non-string acceptance_criterion in engine config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { engine: { acceptance_criterion: 1 } },
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('accepts supported candidate_selection_strategy in engine config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { engine: { candidate_selection_strategy: 'top_k_pareto' } },
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects unsupported candidate_selection_strategy in engine config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { engine: { candidate_selection_strategy: 'greedy' } },
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('accepts supported frontier_type in engine config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { engine: { frontier_type: 'cartesian' } },
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects unsupported frontier_type in engine config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { engine: { frontier_type: 'unknown' } },
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('accepts supported module_selector and batch_sampler in reflection config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { reflection: { module_selector: 'all', batch_sampler: 'epoch_shuffled' } },
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects unsupported module_selector and batch_sampler in reflection config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { reflection: { module_selector: 'unknown', batch_sampler: 'unknown' } },
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('accepts skip_perfect_score and perfect_score in reflection config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { reflection: { skip_perfect_score: true, perfect_score: 1 } },
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects invalid perfect-score reflection config types', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { reflection: { skip_perfect_score: 'true', perfect_score: '1' } },
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('accepts refiner_lm_handle and max_refinements in refiner config', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { refiner: { refiner_lm_handle: 'refiner-1', max_refinements: 2 } },
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects invalid refiner config handle and max_refinements types', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { refiner: { refiner_lm_handle: 42, max_refinements: '2' } },
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('accepts merge config values', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { merge: { max_merge_invocations: 2, merge_val_overlap_floor: 3 } },
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects invalid merge config values', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: { merge: { max_merge_invocations: '2', merge_val_overlap_floor: '3' } },
      },
    };
    expect(is_optimize_request(req)).toBe(false);
  });

  it('accepts tracking config values', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: {
          tracking: {
            use_wandb: true,
            wandb_api_key: 'key',
            wandb_init_kwargs: { project: 'gepa' },
            wandb_attach_existing: false,
            wandb_step_metric: 'iteration',
            use_mlflow: true,
            mlflow_tracking_uri: 'file:///tmp/mlruns',
            mlflow_experiment_name: 'exp',
            mlflow_attach_existing: true,
            key_prefix: 'gepa/',
          },
        },
      },
    };
    expect(is_optimize_request(req)).toBe(true);
  });

  it('rejects invalid tracking config values', () => {
    const req = {
      ...BASE_REQ,
      params: {
        ...BASE_PARAMS,
        config: {
          tracking: {
            use_wandb: 'true',
            wandb_init_kwargs: [],
            mlflow_attach_existing: 'false',
          },
        },
      },
    };
    expect(is_optimize_request(req)).toBe(false);
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
