import { describe, expect, test } from 'vitest';
import { optimize_anything } from '../../src/index.js';
import type { Evaluator } from '../../src/types.js';

describe('engine opt_state forwarding', () => {
  test('passes best_example_evals from prior calls in score-desc order', async () => {
    const seen_ctxs: Array<{ opt_state_scores: number[] }> = [];
    const scores = [0.2, 0.6, 0.4, 0.1];
    let call_idx = 0;

    const evaluator: Evaluator = (_candidate, ctx) => {
      const opt_scores = ctx?.opt_state?.best_example_evals.map((x) => x.score) ?? [];
      seen_ctxs.push({ opt_state_scores: opt_scores });
      const score = scores[call_idx] ?? 0;
      call_idx += 1;
      return [score, { call_idx }];
    };

    await optimize_anything({
      seed_candidate: 'seed',
      evaluator,
      config: {
        engine: { max_metric_calls: 3, best_example_evals_k: 30 },
        reflection: { reflection_lm: async () => '```\nseed\n```' },
      },
    });

    expect(seen_ctxs.length).toBeGreaterThanOrEqual(3);
    expect(seen_ctxs[0]?.opt_state_scores).toEqual([]);
    expect(seen_ctxs[1]?.opt_state_scores).toEqual([0.2]);
    expect(seen_ctxs[2]?.opt_state_scores).toEqual([0.6, 0.2]);
  });
});
