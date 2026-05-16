import { describe, expect, it } from 'vitest';
import { GEPAState, ValsetEvaluation } from '../../src/state.js';
import { result_from_dict, result_from_state } from '../../src/result.js';
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

  it('serializes with upstream schema version 2', () => {
    const state = make_two_program_state();
    const result = result_from_state(state, { run_dir: '/tmp/gepa', seed: 42 });

    const serialized = result.to_dict();

    expect(serialized['validation_schema_version']).toBe(2);
    expect(serialized['candidates']).toEqual(result.candidates);
    expect(serialized['parents']).toEqual(result.parents);
    expect(serialized['best_idx']).toBe(result.best_idx);
    expect(serialized['run_dir']).toBe('/tmp/gepa');
    expect(serialized['seed']).toBe(42);
  });

  it('renders candidate tree visualizations from result data', () => {
    const state = make_two_program_state();
    const result = result_from_state(state, {});

    expect(result.candidate_tree_dot()).toContain('digraph G');
    expect(result.candidate_tree_dot()).toContain('0 -> 1');
    expect(result.candidate_tree_html()).toContain('<!DOCTYPE html>');
  });
});

describe('result_from_dict', () => {
  it('upcasts upstream version 0 payloads', () => {
    const result = result_from_dict({
      candidates: [{ system_prompt: 'weight=0' }, { system_prompt: 'weight=1' }],
      parents: [[null], [0]],
      val_aggregate_scores: [0.15, 0.35],
      val_subscores: [[0.1, 0.2], [0.3, 0.4]],
      per_val_instance_best_candidates: [[0], [1]],
      discovery_eval_counts: [0, 2],
      best_outputs_valset: [
        [[0, { value: 0.1 }]],
        [[1, { value: 0.4 }]],
      ],
      total_metric_calls: 5,
      num_full_val_evals: 2,
      run_dir: '/tmp/gepa',
      seed: 42,
    });

    expect(result.val_subscores).toEqual([
      new Map([[0, 0.1], [1, 0.2]]),
      new Map([[0, 0.3], [1, 0.4]]),
    ]);
    expect(result.per_val_instance_best_candidates).toEqual(new Map([
      [0, new Set([0])],
      [1, new Set([1])],
    ]));
    expect(result.best_outputs_valset).toEqual(new Map([
      [0, [[0, { value: 0.1 }]]],
      [1, [[1, { value: 0.4 }]]],
    ]));
    expect(result.to_dict()['validation_schema_version']).toBe(2);
  });

  it('roundtrips version 2 payloads', () => {
    const result = result_from_dict({
      validation_schema_version: 2,
      candidates: [{ system_prompt: 'a' }, { system_prompt: 'b' }],
      parents: [[null], [0]],
      val_aggregate_scores: [0.4, 0.8],
      val_subscores: [{ '0': 0.2, '1': 0.6 }, { '0': 0.9, '1': 0.7 }],
      per_val_instance_best_candidates: { '0': [1], '1': [1] },
      discovery_eval_counts: [0, 3],
      best_outputs_valset: { '0': [[1, 'out']] },
      val_aggregate_subscores: [{ exact: 0.2 }, { exact: 0.9 }],
      per_objective_best_candidates: { exact: [1] },
      objective_pareto_front: { exact: 0.9 },
      total_metric_calls: 4,
      num_full_val_evals: 2,
      run_dir: null,
      seed: 0,
    });

    expect(result.best_idx).toBe(1);
    expect(result.val_subscores[0]).toEqual(new Map([[0, 0.2], [1, 0.6]]));
    expect(result.per_objective_best_candidates).toEqual(new Map([['exact', new Set([1])]]));
    expect(result.objective_pareto_front).toEqual({ exact: 0.9 });
    expect(result.to_dict()['best_outputs_valset']).toEqual({ '0': [[1, 'out']] });
  });

  it('rejects unsupported future schema versions', () => {
    expect(() => result_from_dict({ validation_schema_version: 3 })).toThrow(
      'Unsupported GEPAResult validation schema version 3',
    );
  });
});
