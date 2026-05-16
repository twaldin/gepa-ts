import { describe, it, expect } from 'vitest';
import {
  DEFAULT_REFINER_PROMPT,
  _build_seed_generation_prompt,
  _generate_seed_candidate,
  optimize_anything,
} from '../../src/index.js';
import { STR_CANDIDATE_KEY } from '../../src/types.js';
import type {
  BatchSampler,
  Candidate,
  CandidateSelector,
  GEPAAdapter,
  GEPAStateLike,
  LanguageModel,
  ReflectionComponentSelector,
} from '../../src/types.js';
import { ExperimentTracker } from '../../src/logging/index.js';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';

describe('optimize_anything reflection_lm validation', () => {
  it('throws when reflection_lm is missing', async () => {
    await expect(optimize_anything({
      seed_candidate: 'x',
      evaluator: () => [0.5, {}],
      config: { engine: { max_metric_calls: 3 }, reflection: {} },
    })).rejects.toThrow(/reflection_lm/);
  });

  it('turns string reflection_lm names into injectable LM wrappers', async () => {
    const completion = () => '```\ny\n```';

    const result = await optimize_anything({
      seed_candidate: 'x',
      evaluator: (candidate) => String(candidate) === 'y' ? 1 : 0.5,
      config: {
        engine: { max_metric_calls: 3 },
        reflection: {
          reflection_lm: 'openai/test-model',
          reflection_lm_kwargs: { completion },
          reflection_minibatch_size: 1,
        },
      },
    });

    expect(result.best_candidate).toBe('y');
  });

  it('throws an actionable zero-dependency LM error for string reflection_lm without a hook', async () => {
    await expect(optimize_anything({
      seed_candidate: 'x',
      evaluator: () => [0.5, {}],
      config: { engine: { max_metric_calls: 3 }, reflection: { reflection_lm: 'not-a-function' } },
    })).rejects.toThrow(/injected completion hook/);
  });
});

describe('seedless candidate generation', () => {
  it('builds the upstream-shaped seed generation prompt', () => {
    const prompt = _build_seed_generation_prompt({
      objective: 'Solve problems.',
      background: 'Use terse answers.',
      dataset: [{ input: 'a' }, { input: 'b' }, { input: 'c' }, { input: 'd' }],
    });

    expect(prompt).toContain('## Goal');
    expect(prompt).toContain('Solve problems.');
    expect(prompt).toContain('## Domain Context & Constraints');
    expect(prompt).toContain('Use terse answers.');
    expect(prompt).toContain('## Sample Inputs');
    expect(prompt).toContain('Example 1');
    expect(prompt).toContain('Example 3');
    expect(prompt).not.toContain('Example 4');
    expect(prompt).toContain('``` blocks');
  });

  it('extracts seed candidate text from fenced LM output', async () => {
    const calls: string[] = [];
    const seed = await _generate_seed_candidate({
      lm: async (prompt) => {
        calls.push(prompt);
        return '```python\ndef solve():\n    return 42\n```';
      },
      objective: 'Write code.',
    });

    expect(calls[0]).toContain('Write code.');
    expect(seed).toEqual({ [STR_CANDIDATE_KEY]: 'def solve():\n    return 42' });
  });

  it('uses generated text as a string candidate when seed_candidate is null', async () => {
    const candidates: string[] = [];
    const result = await optimize_anything({
      seed_candidate: null,
      evaluator: (candidate) => {
        candidates.push(String(candidate));
        return [String(candidate).length, {}];
      },
      objective: 'Generate a candidate.',
      config: {
        engine: { max_metric_calls: 1 },
        reflection: {
          reflection_lm: async () => '```\ngenerated initial candidate\n```',
          reflection_minibatch_size: 1,
        },
      },
    });

    expect(candidates[0]).toBe('generated initial candidate');
    expect(result.best_candidate).toBe('generated initial candidate');
  });
});

describe('refiner configuration plumbing', () => {
  it('injects the default refiner prompt without overwriting user candidates', async () => {
    const result = await optimize_anything({
      seed_candidate: { number: '50' },
      evaluator: (candidate) => {
        const obj = candidate as Record<string, string>;
        return [obj.refiner_prompt?.includes('Guess the number') ? 1 : 0, {}];
      },
      objective: 'Guess the number.',
      config: {
        engine: { max_metric_calls: 1 },
        reflection: { reflection_lm: async () => '```\n51\n```' },
        refiner: { max_refinements: 1 },
      },
    });

    expect(result.best_candidate).toMatchObject({
      number: '50',
      refiner_prompt: expect.stringContaining('Guess the number.'),
    });
    expect(DEFAULT_REFINER_PROMPT).toContain('{objective}');
  });

  it('uses reflection_lm as the default refiner_lm and counts refinement evaluations', async () => {
    let evaluatorCalls = 0;
    const result = await optimize_anything({
      seed_candidate: { number: '10' },
      evaluator: (candidate) => {
        evaluatorCalls += 1;
        const obj = candidate as Record<string, string>;
        return [obj.number === '42' ? 1 : 0, { scores: { accuracy: obj.number === '42' ? 1 : 0 } }];
      },
      objective: 'Guess the number.',
      config: {
        engine: { max_metric_calls: 1 },
        reflection: { reflection_lm: async () => '```json\n{"number":"42"}\n```' },
        refiner: { max_refinements: 1 },
      },
    });

    expect(evaluatorCalls).toBe(2);
    expect(result.total_metric_calls).toBe(2);
    expect(result.val_aggregate_scores[result.best_idx]).toBe(1);
    expect(result.best_candidate).toMatchObject({ number: '10', refiner_prompt: expect.any(String) });
  });
});

describe('cache evaluation storage plumbing', () => {
  it('logs configs, metrics, tables, and final summary to the configured experiment tracker', async () => {
    const tracker = new ExperimentTracker();
    const result = await optimize_anything({
      seed_candidate: { instructions: 'seed' },
      dataset: ['a', 'bb'],
      evaluator: (candidate) => {
        const prompt = (candidate as Record<string, string>).instructions;
        return [prompt.includes('better') ? 1 : 0, { scores: { quality: prompt.includes('better') ? 1 : 0 } }];
      },
      config: {
        engine: {
          max_metric_calls: 20,
          max_candidate_proposals: 1,
          acceptance_criterion: 'improvement_or_equal',
        },
        reflection: {
          reflection_minibatch_size: 1,
          reflection_lm: async () => '```\nbetter instructions\n```',
        },
        tracking: {
          experiment_tracker: tracker,
        },
      },
    });

    expect(result.num_candidates).toBe(2);
    expect(tracker.logged_configs).toMatchObject({ seed: 0, trainset_size: 2, valset_size: 2 });
    expect(tracker.logged_metrics.some((entry) => entry.metrics.new_program_idx === 0)).toBe(true);
    expect(tracker.logged_metrics.some((entry) => entry.metrics.new_program_idx === 1 && entry.metrics['objective/quality'] === 1)).toBe(true);
    expect(tracker.logged_tables.map((table) => table.key)).toContain('candidates');
    expect(tracker.logged_tables.map((table) => table.key)).toContain('valset_scores');
    expect(tracker.logged_tables.map((table) => table.key)).toContain('proposals');
    const proposals_table = tracker.logged_tables.find((table) => table.key === 'proposals');
    if (proposals_table === undefined) {
      throw new Error('expected proposals table');
    }
    expect(proposals_table.columns).toEqual([
      'iteration',
      'component',
      'status',
      'candidate_idx',
      'parent_ids',
      'subsample_score_before',
      'subsample_score_after',
      'prompt',
      'raw_lm_output',
      'proposed_text',
    ]);
    expect(proposals_table.data).toHaveLength(1);
    expect(proposals_table.data[0]?.slice(0, 7)).toEqual([1, 'instructions', 'accepted', 1, '[0]', 0, 1]);
    expect(proposals_table.data[0]?.[7]).toEqual(expect.stringContaining('seed'));
    expect(proposals_table.data[0]?.[8]).toBe('```\nbetter instructions\n```');
    expect(proposals_table.data[0]?.[9]).toBe('better instructions');
    expect(tracker.logged_summaries).toEqual(expect.arrayContaining([
      { key: 'best_candidate_idx', value: 1 },
      { key: 'total_candidates', value: 2 },
      { key: 'best/instructions', value: 'better instructions' },
    ]));
  });

  it('creates disk cache directory for auto mode when run_dir is provided', async () => {
    const runDir = mkdtempSync(`${tmpdir()}/gepa-ts-cache-`);
    try {
      await optimize_anything({
        seed_candidate: 'x',
        evaluator: () => [0.5, {}],
        config: {
          engine: { max_metric_calls: 1, cache_evaluation: true, cache_evaluation_storage: 'auto', run_dir: runDir },
          reflection: { reflection_lm: async () => '```\nx\n```' },
        },
      });

      expect(existsSync(`${runDir}/fitness_cache`)).toBe(true);
    } finally {
      rmSync(runDir, { recursive: true, force: true });
    }
  });

  it('rejects disk cache mode without run_dir', async () => {
    await expect(optimize_anything({
      seed_candidate: 'x',
      evaluator: () => [0.5, {}],
      config: {
        engine: { max_metric_calls: 1, cache_evaluation: true, cache_evaluation_storage: 'disk' },
        reflection: { reflection_lm: async () => '```\nx\n```' },
      },
    })).rejects.toThrow(/cache_evaluation_storage='disk' requires run_dir/);
  });

  it('uses cached valset results without spending metric calls on repeated candidate full evals', async () => {
    let metricCalls = 0;
    const result = await optimize_anything({
      seed_candidate: 'x',
      dataset: ['a', 'bb'],
      evaluator: () => {
        metricCalls += 1;
        return [0.5, {}];
      },
      config: {
        engine: {
          max_metric_calls: 100,
          max_candidate_proposals: 1,
          acceptance_criterion: 'improvement_or_equal',
          cache_evaluation: true,
        },
        reflection: {
          reflection_minibatch_size: 1,
          custom_candidate_proposer: (candidate) => candidate,
          reflection_lm: async () => 'unused',
        },
      },
    });

    expect(metricCalls).toBe(4);
    expect(result.total_metric_calls).toBe(4);
    expect(result.num_candidates).toBe(2);
  });

  it('resumes optimization state from run_dir', async () => {
    const runDir = mkdtempSync(`${tmpdir()}/gepa-ts-resume-`);
    try {
      const first = await optimize_anything({
        seed_candidate: 'x',
        dataset: ['a', 'bb'],
        evaluator: (candidate, ctx) => [String(candidate).length + String(ctx?.example).length / 10, {}],
        config: {
          engine: { max_metric_calls: 1, run_dir: runDir },
          reflection: { reflection_lm: async () => '```\ny\n```', reflection_minibatch_size: 1 },
        },
      });

      expect(existsSync(`${runDir}/gepa_state.json`)).toBe(true);

      const second = await optimize_anything({
        seed_candidate: 'different',
        dataset: ['a', 'bb'],
        evaluator: (candidate, ctx) => [String(candidate).length + String(ctx?.example).length / 10, {}],
        config: {
          engine: { max_metric_calls: 0, run_dir: runDir },
          reflection: { reflection_lm: async () => '```\nz\n```', reflection_minibatch_size: 1 },
        },
      });

      expect(second.total_metric_calls).toBe(first.total_metric_calls);
      expect(second.best_candidate).toBe(first.best_candidate);
    } finally {
      rmSync(runDir, { recursive: true, force: true });
    }
  });

  it('uses run_dir gepa.stop as an upstream-compatible file stopper', async () => {
    const runDir = mkdtempSync(`${tmpdir()}/gepa-ts-stop-`);
    try {
      writeFileSync(`${runDir}/gepa.stop`, '');
      const result = await optimize_anything({
        seed_candidate: 'x',
        dataset: ['a', 'bb'],
        evaluator: (candidate, ctx) => [String(candidate).length + String(ctx?.example).length / 10, {}],
        config: {
          engine: { run_dir: runDir },
          reflection: { reflection_lm: async () => '```\ny\n```', reflection_minibatch_size: 1 },
        },
      });

      expect(result.total_metric_calls).toBe(2);
      expect(result.num_candidates).toBe(1);
      expect(result.best_candidate).toBe('x');
    } finally {
      rmSync(runDir, { recursive: true, force: true });
    }
  });

  it('honors max_candidate_proposals with upstream state.i semantics', async () => {
    let reflectionCalls = 0;
    const result = await optimize_anything({
      seed_candidate: 'x',
      dataset: ['a'],
      evaluator: () => [0.5, {}],
      config: {
        engine: { max_metric_calls: 100, max_candidate_proposals: 1 },
        reflection: {
          reflection_minibatch_size: 1,
          reflection_lm: async () => {
            reflectionCalls += 1;
            return '```\ny\n```';
          },
        },
      },
    });

    expect(reflectionCalls).toBe(1);
    expect(result.num_candidates).toBe(1);
    expect(result.total_metric_calls).toBe(3);
  });

  it('honors max_reflection_cost using reflection_lm.total_cost', async () => {
    let reflectionCalls = 0;
    const lm: LanguageModel & { total_cost: number } = Object.assign(
      async () => {
        reflectionCalls += 1;
        lm.total_cost = 0.2;
        return '```\ny\n```';
      },
      { total_cost: 0 },
    );

    const result = await optimize_anything({
      seed_candidate: 'x',
      dataset: ['a'],
      evaluator: () => [0.5, {}],
      config: {
        engine: { max_metric_calls: 100, max_reflection_cost: 0.1 },
        reflection: {
          reflection_minibatch_size: 1,
          reflection_lm: lm,
        },
      },
    });

    expect(reflectionCalls).toBe(1);
    expect(result.num_candidates).toBe(1);
    expect(result.total_metric_calls).toBe(3);
  });

  it('tracks best outputs when track_best_outputs is enabled', async () => {
    const result = await optimize_anything({
      seed_candidate: 'x',
      dataset: ['a', 'bb'],
      evaluator: (candidate, ctx) => [
        String(ctx?.example).length,
        { output: `${String(candidate)}:${String(ctx?.example)}` },
      ],
      config: {
        engine: { max_metric_calls: 1, track_best_outputs: true },
        reflection: { reflection_lm: async () => '```\ny\n```', reflection_minibatch_size: 1 },
      },
    });

    expect(result.best_outputs_valset?.get(0)).toEqual([[0, [1, { current_candidate: 'x' }, { output: 'x:a' }]]]);
    expect(result.best_outputs_valset?.get(1)).toEqual([[0, [2, { current_candidate: 'x' }, { output: 'x:bb' }]]]);
  });

  it('accepts equal-score proposals with improvement_or_equal acceptance criterion', async () => {
    const result = await optimize_anything({
      seed_candidate: 'x',
      dataset: ['a'],
      evaluator: () => [0.5, {}],
      config: {
        engine: { max_metric_calls: 3, acceptance_criterion: 'improvement_or_equal' },
        reflection: { reflection_lm: async () => '```\ny\n```', reflection_minibatch_size: 1 },
      },
    });

    expect(result.num_candidates).toBe(2);
    expect(result.candidates[1]).toEqual({ [STR_CANDIDATE_KEY]: 'y' });
  });

  it('updates all candidate components with module_selector all', async () => {
    const result = await optimize_anything({
      seed_candidate: { first: 'x', second: 'z' },
      dataset: ['a'],
      evaluator: () => [0.5, {}],
      config: {
        engine: { max_metric_calls: 3, acceptance_criterion: 'improvement_or_equal' },
        reflection: {
          module_selector: 'all',
          reflection_lm: async () => '```\ny\n```',
          reflection_minibatch_size: 1,
        },
      },
    });

    expect(result.candidates[1]).toEqual({ first: 'y', second: 'y' });
  });

  it('uses a custom module_selector instance', async () => {
    const selected: string[][] = [];
    const custom_selector: ReflectionComponentSelector = (_state, _trajectories, _scores, _idx, candidate) => {
      selected.push(Object.keys(candidate));
      return ['second'];
    };

    const result = await optimize_anything({
      seed_candidate: { first: 'x', second: 'x' },
      dataset: ['a'],
      evaluator: (candidate) => {
        const c = candidate as Record<string, string>;
        return [c.second === 'y' ? 1 : 0.5, {}];
      },
      config: {
        engine: { max_metric_calls: 4, acceptance_criterion: 'improvement_or_equal' },
        reflection: {
          reflection_lm: async () => '```\ny\n```',
          reflection_minibatch_size: 1,
          module_selector: custom_selector,
        },
      },
    });

    expect(selected).toEqual([['first', 'second']]);
    expect(result.best_candidate).toMatchObject({ first: 'x', second: 'y' });
  });

  it('uses a custom batch_sampler instance', async () => {
    const sampled_states: number[] = [];
    const custom_sampler: BatchSampler<number, string> = {
      next_minibatch_ids(_loader, state: GEPAStateLike) {
        sampled_states.push(state.total_num_evals);
        return [1];
      },
    };
    const seen_examples: unknown[] = [];

    await optimize_anything({
      seed_candidate: 'x',
      dataset: ['first', 'second'],
      evaluator: (_candidate, ctx) => {
        seen_examples.push(ctx?.example);
        return [0.5, {}];
      },
      config: {
        engine: { max_metric_calls: 4 },
        reflection: {
          reflection_lm: async () => '```\ny\n```',
          batch_sampler: custom_sampler,
        },
      },
    });

    expect(sampled_states).toEqual([2]);
    expect(seen_examples).toContain('second');
  });

  it('rejects reflection_minibatch_size with a custom batch_sampler', async () => {
    const custom_sampler: BatchSampler<number, string> = {
      next_minibatch_ids() {
        return [0];
      },
    };

    await expect(optimize_anything({
      seed_candidate: 'x',
      dataset: ['a'],
      evaluator: () => [0.5, {}],
      config: {
        engine: { max_metric_calls: 3 },
        reflection: {
          reflection_lm: async () => '```\ny\n```',
          batch_sampler: custom_sampler,
          reflection_minibatch_size: 1,
        },
      },
    })).rejects.toThrow(/reflection_minibatch_size/);
  });

  it('can optimize with a BYO adapter instead of an evaluator', async () => {
    const adapter: GEPAAdapter<string, Record<string, unknown>, string> = {
      evaluate: async (batch, candidate, capture_traces = false) => ({
        outputs: batch.map((example) => `${candidate.instructions}:${example}`),
        scores: batch.map(() => 0.5),
        ...(capture_traces ? { trajectories: batch.map((example) => ({ example })) } : {}),
        num_metric_calls: batch.length,
      }),
      make_reflective_dataset: (_candidate, eval_batch, components_to_update) => {
        const ret: Record<string, Array<Record<string, unknown>>> = {};
        for (const component of components_to_update) {
          ret[component] = (eval_batch.trajectories ?? []).map((trajectory) => ({
            Inputs: trajectory,
            'Generated Outputs': 'old',
            Feedback: 'improve',
          }));
        }
        return ret;
      },
    };

    const result = await optimize_anything({
      seed_candidate: { instructions: 'x' },
      adapter,
      dataset: ['a'],
      config: {
        engine: { max_metric_calls: 3, acceptance_criterion: 'improvement_or_equal' },
        reflection: { reflection_lm: async () => '```\ny\n```', reflection_minibatch_size: 1 },
      },
    });

    expect(result.num_candidates).toBe(2);
    expect(result.candidates[1]).toEqual({ instructions: 'y' });
  });

  it('wires merge config into native merge proposals before reflective mutation', async () => {
    const always_seed_selector: CandidateSelector = {
      select_candidate_idx: () => 0,
    };
    const lm_outputs = ['```\nleft\n```', '```\nright\n```'];
    const events: string[] = [];

    const result = await optimize_anything({
      seed_candidate: { first: 'base', second: 'base' },
      dataset: [{ id: 0 }, { id: 1 }],
      evaluator: (candidate, ctx) => {
        const c = candidate as Candidate;
        const example = ctx?.example;
        const id = typeof example === 'object' && example !== null && 'id' in example
          ? example.id
          : null;
        const score = id === 0
          ? (c.first === 'left' ? 1 : 0.1)
          : (id === 1 && c.second === 'right' ? 1 : 0.1);
        return [score, { scores: { target: score } }];
      },
      config: {
        engine: {
          max_metric_calls: 40,
          max_candidate_proposals: 3,
          candidate_selection_strategy: always_seed_selector,
        },
        reflection: {
          reflection_lm: async () => lm_outputs.shift() ?? '```\nunused\n```',
          reflection_minibatch_size: 2,
        },
        merge: {
          max_merge_invocations: 1,
          merge_val_overlap_floor: 2,
        },
        callbacks: [{
          on_merge_attempted: () => events.push('attempted'),
          on_merge_accepted: () => events.push('accepted'),
        }],
      },
    });

    expect(result.best_candidate).toEqual({ first: 'left', second: 'right' });
    expect(result.parents[result.best_idx]).toEqual([1, 2]);
    expect(events).toEqual(['attempted', 'accepted']);
  });
});
