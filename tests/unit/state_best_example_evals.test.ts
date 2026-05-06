import { describe, expect, test } from 'vitest';
import { GEPAState, ValsetEvaluation } from '../../src/state';

describe('GEPAState best_example_evals', () => {
  test('record_example_eval keeps top-k scores sorted desc', () => {
    const state = new GEPAState({
      seed_candidate: { prompt: 'x' },
      base_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.1]]),
        outputs_by_val_id: new Map([[0, 'out']]),
      }),
    });

    for (const score of [0.4, 0.1, 0.9, 0.7, 0.2]) {
      state.record_example_eval(123, score, { s: score }, 3);
    }

    const best = state.best_example_evals.get(123);
    expect(best).toBeDefined();
    expect(best).toHaveLength(3);
    expect(best?.map((x) => x.score)).toEqual([0.9, 0.7, 0.4]);
  });
});
