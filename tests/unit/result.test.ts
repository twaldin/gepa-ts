import { describe, expect, it } from 'vitest';
import { GEPAState, ValsetEvaluation } from '../../src/state.js';
import { result_from_state } from '../../src/result.js';
import { STR_CANDIDATE_KEY } from '../../src/types.js';

function make_two_program_state(): GEPAState {
  const state = new GEPAState({
    seed_candidate: { text: 'first' },
    base_evaluation: new ValsetEvaluation({
      outputs_by_val_id: new Map([
        [0, null],
        [1, null],
      ]),
      scores_by_val_id: new Map([
        [0, 0.4],
        [1, 0.5],
      ]),
    }),
  });

  state.update_state_with_new_program({
    parent_program_idx: [0],
    new_program: { text: 'second' },
    valset_evaluation: new ValsetEvaluation({
      outputs_by_val_id: new Map([
        [0, null],
        [1, null],
      ]),
      scores_by_val_id: new Map([
        [0, 0.7],
        [1, 0.8],
      ]),
    }),
    run_dir: null,
    num_metric_calls_by_discovery_of_new_program: 5,
  });

  return state;
}

describe('result_from_state', () => {
  it('produces candidates and scores matching state', () => {
    const state = make_two_program_state();
    const result = result_from_state(state, {});

    expect(result.candidates).toHaveLength(2);
    expect(result.val_aggregate_scores).toHaveLength(2);
    expect(result.parents).toEqual(state.parent_program_for_candidate);

    for (const [val_id, front] of state.program_at_pareto_front_valset) {
      const result_front = result.per_val_instance_best_candidates.get(val_id);
      expect(result_front).toBeDefined();
      expect(result_front).toEqual(front);
    }
  });

  it('best_idx returns argmax of val_aggregate_scores', () => {
    const state = make_two_program_state();
    const result = result_from_state(state, {});

    const scores = result.val_aggregate_scores;
    const expected_best = scores.reduce((best_i, s, i) => (s > (scores[best_i] ?? -Infinity) ? i : best_i), 0);
    expect(result.best_idx).toBe(expected_best);

    const max_score = Math.max(...scores);
    expect(scores[result.best_idx]).toBe(max_score);
  });

  it('best_candidate unwraps str when _str_candidate_key set', () => {
    const state = new GEPAState({
      seed_candidate: { [STR_CANDIDATE_KEY]: 'my prompt' },
      base_evaluation: new ValsetEvaluation({
        outputs_by_val_id: new Map([[0, null]]),
        scores_by_val_id: new Map([[0, 0.5]]),
      }),
    });

    const result = result_from_state(state, { str_candidate_key: STR_CANDIDATE_KEY });

    expect(typeof result.best_candidate).toBe('string');
    expect(result.best_candidate).toBe('my prompt');
  });

  it('best_candidate returns full dict when _str_candidate_key not set', () => {
    const state = new GEPAState({
      seed_candidate: { text: 'hello' },
      base_evaluation: new ValsetEvaluation({
        outputs_by_val_id: new Map([[0, null]]),
        scores_by_val_id: new Map([[0, 0.5]]),
      }),
    });

    const result = result_from_state(state, {});

    expect(typeof result.best_candidate).toBe('object');
    expect(result.best_candidate).toEqual({ text: 'hello' });
  });
});
