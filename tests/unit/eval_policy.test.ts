import { describe, expect, test } from "vitest";
import { FullEvaluationPolicy } from "../../src/eval_policy";
import { GEPAState, ValsetEvaluation } from "../../src/state";

function make_state(): GEPAState<unknown, number> {
  const seed_eval = new ValsetEvaluation({
    outputs_by_val_id: new Map([
      [0, "a"],
      [1, "b"],
      [2, "c"],
    ]),
    scores_by_val_id: new Map([
      [0, 0.6],
      [1, 0.2],
      [2, 0.4],
    ]),
  });
  return new GEPAState({
    seed_candidate: { p: "seed" },
    base_evaluation: seed_eval,
  });
}

describe("FullEvaluationPolicy", () => {
  test("get_eval_batch returns all ids", () => {
    const policy = new FullEvaluationPolicy<number, string>();
    const loader = {
      all_ids: () => [0, 1, 2],
      fetch: (ids: number[]) => ids.map(String),
      length: 3,
    };
    expect(policy.get_eval_batch(loader, { total_num_evals: 0 })).toEqual([0, 1, 2]);
  });

  test("get_best_program uses average then coverage", () => {
    const state = make_state();
    state.update_state_with_new_program({
      parent_program_idx: [0],
      new_program: { p: "p1" },
      valset_evaluation: new ValsetEvaluation({
        outputs_by_val_id: new Map([
          [0, "x"],
          [1, "y"],
        ]),
        scores_by_val_id: new Map([
          [0, 1],
          [1, 1],
        ]),
      }),
      run_dir: null,
      num_metric_calls_by_discovery_of_new_program: 0,
    });
    state.update_state_with_new_program({
      parent_program_idx: [0],
      new_program: { p: "p2" },
      valset_evaluation: new ValsetEvaluation({
        outputs_by_val_id: new Map([
          [0, "m"],
          [1, "n"],
          [2, "o"],
        ]),
        scores_by_val_id: new Map([
          [0, 1],
          [1, 1],
          [2, 1],
        ]),
      }),
      run_dir: null,
      num_metric_calls_by_discovery_of_new_program: 0,
    });

    const policy = new FullEvaluationPolicy<number, string>();
    expect(policy.get_best_program(state)).toBe(2);
  });

  test("get_valset_score matches state average", () => {
    const state = make_state();
    const policy = new FullEvaluationPolicy<number, string>();
    expect(policy.get_valset_score(0, state)).toBe(state.get_program_average_val_subset(0)[0]);
  });
});
