import { describe, expect, test, vi } from "vitest";
import { GEPAState, ValsetEvaluation, initialize_gepa_state } from "../../src/state";

describe("GEPAState", () => {
  test("initialize_gepa_state seeds state fields", () => {
    const state = initialize_gepa_state({
      run_dir: null,
      logger: { log: () => {} },
      seed_candidate: { prompt: "x" },
      seed_valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.5]]),
        outputs_by_val_id: new Map([[0, "out0"]]),
      }),
    });

    expect(state.program_candidates.length).toBe(1);
    expect(state.pareto_front_valset.get(0)).toBe(0.5);
    expect(state.list_of_named_predictors).toEqual(["prompt"]);
    expect(state.i).toBe(-1);
    expect(state.total_num_evals).toBe(1);
    expect(state.num_full_ds_evals).toBe(1);
  });

  test("update_state_with_new_program replaces val-id front on strict improvement", () => {
    const state = initialize_gepa_state({
      run_dir: null,
      logger: { log: () => {} },
      seed_candidate: { prompt: "x" },
      seed_valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.5]]),
        outputs_by_val_id: new Map([[0, "out0"]]),
      }),
    });

    state.update_state_with_new_program({
      parent_program_idx: [0],
      new_program: { prompt: "y" },
      valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.9]]),
        outputs_by_val_id: new Map([[0, "out1"]]),
      }),
      run_dir: null,
      num_metric_calls_by_discovery_of_new_program: 1,
    });

    expect(state.pareto_front_valset.get(0)).toBe(0.9);
    expect(state.program_at_pareto_front_valset.get(0)).toEqual(new Set([1]));
  });

  test("equal score keeps both programs on val-id front", () => {
    const state = initialize_gepa_state({
      run_dir: null,
      logger: { log: () => {} },
      seed_candidate: { prompt: "x" },
      seed_valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.5]]),
        outputs_by_val_id: new Map([[0, "out0"]]),
      }),
    });

    state.update_state_with_new_program({
      parent_program_idx: [0],
      new_program: { prompt: "y" },
      valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.5]]),
        outputs_by_val_id: new Map([[0, "out1"]]),
      }),
      run_dir: null,
      num_metric_calls_by_discovery_of_new_program: 1,
    });

    expect(state.program_at_pareto_front_valset.get(0)).toEqual(new Set([0, 1]));
  });

  test("program_full_scores_val_set averages per-program subscores", () => {
    const state = initialize_gepa_state({
      run_dir: null,
      logger: { log: () => {} },
      seed_candidate: { prompt: "x" },
      seed_valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([
          [0, 0.5],
          [1, 1.0],
        ]),
        outputs_by_val_id: new Map([
          [0, "out0"],
          [1, "out1"],
        ]),
      }),
    });

    state.update_state_with_new_program({
      parent_program_idx: [0],
      new_program: { prompt: "y" },
      valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([
          [0, 0.4],
          [1, 0.8],
        ]),
        outputs_by_val_id: new Map([
          [0, "out2"],
          [1, "out3"],
        ]),
      }),
      run_dir: null,
      num_metric_calls_by_discovery_of_new_program: 1,
    });

    expect(state.program_full_scores_val_set[0]).toBeCloseTo(0.75);
    expect(state.program_full_scores_val_set[1]).toBeCloseTo(0.6);
  });

  test("frontier_type objective throws in v1", () => {
    expect(
      () =>
        new GEPAState({
          seed_candidate: { prompt: "x" },
          base_evaluation: new ValsetEvaluation({
            scores_by_val_id: new Map([[0, 1]]),
            outputs_by_val_id: new Map([[0, "o"]]),
          }),
          frontier_type: "objective",
        }),
    ).toThrow('frontier_type "objective" not supported in v1');
  });

  test("is_consistent true after initialization", () => {
    const state = initialize_gepa_state({
      run_dir: null,
      logger: { log: () => {} },
      seed_candidate: { prompt: "x" },
      seed_valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.5]]),
        outputs_by_val_id: new Map([[0, "out0"]]),
      }),
    });

    expect(state.is_consistent()).toBe(true);
  });

  test("add_budget_hook receives new_total and delta", () => {
    const state = initialize_gepa_state({
      run_dir: null,
      logger: { log: () => {} },
      seed_candidate: { prompt: "x" },
      seed_valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.5]]),
        outputs_by_val_id: new Map([[0, "out0"]]),
      }),
    });

    const hook = vi.fn();
    state.add_budget_hook(hook);
    state.increment_evals(3);

    expect(hook).toHaveBeenCalledWith(4, 3);
  });

  test("_aggregate_objective_scores(null) returns empty object", () => {
    expect(GEPAState._aggregate_objective_scores(null)).toEqual({});
  });

  test("_aggregate_objective_scores averages objective values", () => {
    const result = GEPAState._aggregate_objective_scores(
      new Map([
        [0, { acc: 0.5, lat: 0.7 }],
        [1, { acc: 0.3 }],
      ]),
    );

    expect(result).toEqual({ acc: 0.4, lat: 0.7 });
  });

  test("get_pareto_front_mapping returns instance mapping", () => {
    const state = initialize_gepa_state({
      run_dir: null,
      logger: { log: () => {} },
      seed_candidate: { prompt: "x" },
      seed_valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.5]]),
        outputs_by_val_id: new Map([[0, "out0"]]),
      }),
    });

    expect(state.get_pareto_front_mapping()).toEqual(new Map([[0, new Set([0])]]));
  });

  test("cached_evaluate_full without cache evaluates and packages maps", () => {
    const state = initialize_gepa_state({
      run_dir: null,
      logger: { log: () => {} },
      seed_candidate: { prompt: "x" },
      seed_valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.5]]),
        outputs_by_val_id: new Map([[0, "out0"]]),
      }),
    });

    const fetcher = (ids: number[]) => ids.map((id) => ({ id }));
    const evaluator = (
      batch: Array<{ id: number }>,
      _candidate: Record<string, string>,
    ): [string[], number[], Array<Record<string, number>>] => {
      return [
        batch.map((x) => `o${x.id}`),
        batch.map((x) => x.id / 10),
        batch.map((x) => ({ acc: x.id / 100 })),
      ];
    };

    const [outputs_by_id, scores_by_id, objective_by_id, eval_count] = state.cached_evaluate_full(
      { prompt: "z" },
      [2, 3],
      fetcher,
      evaluator,
    );

    expect(outputs_by_id).toEqual(new Map([[2, "o2"], [3, "o3"]]));
    expect(scores_by_id).toEqual(new Map([[2, 0.2], [3, 0.3]]));
    expect(objective_by_id).toEqual(new Map([[2, { acc: 0.02 }], [3, { acc: 0.03 }]]));
    expect(eval_count).toBe(2);
  });

  test("cached_evaluate returns scores in example_ids order", () => {
    const state = initialize_gepa_state({
      run_dir: null,
      logger: { log: () => {} },
      seed_candidate: { prompt: "x" },
      seed_valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.5]]),
        outputs_by_val_id: new Map([[0, "out0"]]),
      }),
    });

    const [scores, eval_count] = state.cached_evaluate(
      { prompt: "z" },
      [10, 1],
      (ids: number[]) => ids,
      (batch: number[]) => [
        batch.map((id) => `o${id}`),
        batch.map((id) => id),
        null,
      ],
    );

    expect(scores).toEqual([10, 1]);
    expect(eval_count).toBe(2);
  });
});
