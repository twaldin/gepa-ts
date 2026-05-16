import { describe, expect, test, vi } from "vitest";
import { EvaluationCache, GEPAState, ValsetEvaluation, initialize_gepa_state } from "../../src/state";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";

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

  test("frontier_type objective requires objective scores", () => {
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
    ).toThrow("frontier_type='objective' requires objective_scores");
  });

  test("frontier_type objective tracks objective pareto fronts", () => {
    const state = initialize_gepa_state({
      run_dir: null,
      logger: { log: () => {} },
      seed_candidate: { prompt: "x" },
      seed_valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.5], [1, 0.5]]),
        outputs_by_val_id: new Map([[0, "out0"], [1, "out1"]]),
        objective_scores_by_val_id: new Map([[0, { quality: 0.4, safety: 0.9 }], [1, { quality: 0.6, safety: 0.7 }]]),
      }),
      frontier_type: "objective",
    });

    state.update_state_with_new_program({
      parent_program_idx: [0],
      new_program: { prompt: "y" },
      valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.6], [1, 0.6]]),
        outputs_by_val_id: new Map([[0, "out2"], [1, "out3"]]),
        objective_scores_by_val_id: new Map([[0, { quality: 0.9, safety: 0.5 }], [1, { quality: 0.7, safety: 0.7 }]]),
      }),
      run_dir: null,
      num_metric_calls_by_discovery_of_new_program: 2,
    });

    expect(state.objective_pareto_front).toEqual({ quality: 0.8, safety: 0.8 });
    expect(state.program_at_pareto_front_objectives.get("quality")).toEqual(new Set([1]));
    expect(state.program_at_pareto_front_objectives.get("safety")).toEqual(new Set([0]));
    expect(state.get_pareto_front_mapping()).toEqual(new Map([["quality", new Set([1])], ["safety", new Set([0])]]));
  });

  test("frontier_type hybrid combines instance and objective fronts", () => {
    const state = initialize_gepa_state({
      run_dir: null,
      logger: { log: () => {} },
      seed_candidate: { prompt: "x" },
      seed_valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.5]]),
        outputs_by_val_id: new Map([[0, "out0"]]),
        objective_scores_by_val_id: new Map([[0, { quality: 0.5 }]]),
      }),
      frontier_type: "hybrid",
    });

    const mapping = state.get_pareto_front_mapping();
    expect([...mapping.entries()]).toEqual([
      [["val_id", 0], new Set([0])],
      [["objective", "quality"], new Set([0])],
    ]);
  });

  test("frontier_type cartesian tracks per-example objective fronts", () => {
    const state = initialize_gepa_state({
      run_dir: null,
      logger: { log: () => {} },
      seed_candidate: { prompt: "x" },
      seed_valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.5], [1, 0.5]]),
        outputs_by_val_id: new Map([[0, "out0"], [1, "out1"]]),
        objective_scores_by_val_id: new Map([[0, { quality: 0.4, safety: 0.9 }], [1, { quality: 0.6, safety: 0.7 }]]),
      }),
      frontier_type: "cartesian",
    });

    state.update_state_with_new_program({
      parent_program_idx: [0],
      new_program: { prompt: "y" },
      valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.6], [1, 0.6]]),
        outputs_by_val_id: new Map([[0, "out2"], [1, "out3"]]),
        objective_scores_by_val_id: new Map([[0, { quality: 0.8, safety: 0.8 }], [1, { quality: 0.5, safety: 0.9 }]]),
      }),
      run_dir: null,
      num_metric_calls_by_discovery_of_new_program: 2,
    });

    expect([...state.get_pareto_front_mapping().entries()]).toEqual([
      [["cartesian", 0, "quality"], new Set([1])],
      [["cartesian", 0, "safety"], new Set([0])],
      [["cartesian", 1, "quality"], new Set([0])],
      [["cartesian", 1, "safety"], new Set([1])],
    ]);
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

  test("EvaluationCache stores and retrieves candidate/example evaluations", () => {
    const cache = new EvaluationCache<string, string>();

    expect(cache.get({ prompt: "test" }, "example_1")).toBeNull();

    cache.put({ prompt: "test" }, "example_1", "output1", 0.8, { acc: 0.9 });
    const cached = cache.get({ prompt: "test" }, "example_1");

    expect(cached).not.toBeNull();
    expect(cached?.output).toBe("output1");
    expect(cached?.score).toBe(0.8);
    expect(cached?.objective_scores).toEqual({ acc: 0.9 });
    expect(cache.get({ prompt: "other" }, "example_1")).toBeNull();
  });

  test("EvaluationCache separates examples and batch reports uncached ids", () => {
    const cache = new EvaluationCache<string, string>();
    const candidate = { prompt: "test" };

    cache.put(candidate, "ex1", "out1", 0.5);
    cache.put(candidate, "ex2", "out2", 0.6, { acc: 0.8 });

    const [cached_results, uncached_ids] = cache.get_batch(candidate, ["ex1", "ex2", "ex3"]);

    expect([...cached_results.keys()]).toEqual(["ex1", "ex2"]);
    expect(cached_results.get("ex1")?.output).toBe("out1");
    expect(cached_results.get("ex2")?.objective_scores).toEqual({ acc: 0.8 });
    expect(uncached_ids).toEqual(["ex3"]);
  });

  test("EvaluationCache serialization uses upstream Python candidate hashes", () => {
    const cache = new EvaluationCache<string, string>();

    cache.put({ b: "two", a: "one" }, "ex1", "out1", 0.5);

    expect(cache.serialize()[0]?.candidate_hash).toBe(
      "a5cc796d8f26a92a07e3e475537b32b06b53ee0e54e4d6e1848a32bcc52a8265",
    );
  });

  test("cached_evaluate_full without cache evaluates and packages maps", async () => {
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

    const [outputs_by_id, scores_by_id, objective_by_id, eval_count] = await state.cached_evaluate_full(
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

  test("cached_evaluate returns scores in example_ids order", async () => {
    const state = initialize_gepa_state({
      run_dir: null,
      logger: { log: () => {} },
      seed_candidate: { prompt: "x" },
      seed_valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.5]]),
        outputs_by_val_id: new Map([[0, "out0"]]),
      }),
    });

    const [scores, eval_count] = await state.cached_evaluate(
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

  test("cached_evaluate_full with cache only evaluates misses and preserves order maps", async () => {
    const cache = new EvaluationCache<string, number>();
    cache.put({ prompt: "z" }, 10, "cached10", 10, { acc: 1 });
    const state = initialize_gepa_state<string, number>({
      run_dir: null,
      logger: { log: () => {} },
      seed_candidate: { prompt: "x" },
      seed_valset_evaluation: new ValsetEvaluation({
        scores_by_val_id: new Map([[0, 0.5]]),
        outputs_by_val_id: new Map([[0, "out0"]]),
      }),
      evaluation_cache: cache,
    });

    const fetched: number[][] = [];
    const [outputs_by_id, scores_by_id, objective_by_id, eval_count] = await state.cached_evaluate_full(
      { prompt: "z" },
      [10, 11, 12],
      (ids: number[]) => {
        fetched.push(ids);
        return ids;
      },
      (batch: number[]) => [
        batch.map((id) => `fresh${id}`),
        batch.map((id) => id),
        batch.map((id) => ({ acc: id / 100 })),
      ],
    );

    expect(fetched).toEqual([[11, 12]]);
    expect(eval_count).toBe(2);
    expect(outputs_by_id).toEqual(new Map([[10, "cached10"], [11, "fresh11"], [12, "fresh12"]]));
    expect(scores_by_id).toEqual(new Map([[10, 10], [11, 11], [12, 12]]));
    expect(objective_by_id).toEqual(new Map([[10, { acc: 1 }], [11, { acc: 0.11 }], [12, { acc: 0.12 }]]));
    expect(cache.get({ prompt: "z" }, 11)?.output).toBe("fresh11");
  });

  test("initialize_gepa_state syncs loaded cache with current cache setting", () => {
    const runDir = mkdtempSync(`${tmpdir()}/gepa-ts-cache-state-`);
    try {
      const cache = new EvaluationCache<string, number>();
      cache.put({ prompt: "z" }, 1, "cached", 0.7);
      const state = initialize_gepa_state<string, number>({
        run_dir: runDir,
        logger: { log: () => {} },
        seed_candidate: { prompt: "x" },
        seed_valset_evaluation: new ValsetEvaluation({
          scores_by_val_id: new Map([[1, 0.5]]),
          outputs_by_val_id: new Map([[1, "out1"]]),
        }),
        evaluation_cache: cache,
      });
      state.save(runDir);

      const loaded_with_cache = initialize_gepa_state<string, number>({
        run_dir: runDir,
        logger: { log: () => {} },
        seed_candidate: { prompt: "x" },
        seed_valset_evaluation: new ValsetEvaluation({
          scores_by_val_id: new Map([[1, 0.5]]),
          outputs_by_val_id: new Map([[1, "out1"]]),
        }),
        evaluation_cache: new EvaluationCache<string, number>(),
      });
      expect(loaded_with_cache.evaluation_cache?.get({ prompt: "z" }, 1)?.output).toBe("cached");

      const loaded_without_cache = initialize_gepa_state<string, number>({
        run_dir: runDir,
        logger: { log: () => {} },
        seed_candidate: { prompt: "x" },
        seed_valset_evaluation: new ValsetEvaluation({
          scores_by_val_id: new Map([[1, 0.5]]),
          outputs_by_val_id: new Map([[1, "out1"]]),
        }),
      });
      expect(loaded_without_cache.evaluation_cache).toBeNull();
    } finally {
      rmSync(runDir, { recursive: true, force: true });
    }
  });

  test("save/load round-trips maps, sets, numeric ids, and best eval history", () => {
    const runDir = mkdtempSync(`${tmpdir()}/gepa-ts-state-`);
    try {
      const state = initialize_gepa_state({
        run_dir: runDir,
        logger: { log: () => {} },
        seed_candidate: { prompt: "x" },
        seed_valset_evaluation: new ValsetEvaluation({
          scores_by_val_id: new Map([[1, 0.5], [2, 0.6]]),
          outputs_by_val_id: new Map([[1, { out: "one" }], [2, { out: "two" }]]),
          objective_scores_by_val_id: new Map([[1, { acc: 0.5 }], [2, { acc: 0.7 }]]),
        }),
        track_best_outputs: true,
      });

      state.i = 0;
      state.full_program_trace.push({ i: 0, note: "kept" });
      state.record_example_eval(1, 0.4, { text: "worse" }, 2);
      state.record_example_eval(1, 0.9, { text: "best" }, 2);
      const newIdx = state.update_state_with_new_program({
        parent_program_idx: [0],
        new_program: { prompt: "y" },
        valset_evaluation: new ValsetEvaluation({
          scores_by_val_id: new Map([[1, 0.9], [2, 0.6]]),
          outputs_by_val_id: new Map([[1, { out: "better" }], [2, { out: "tie" }]]),
          objective_scores_by_val_id: new Map([[1, { acc: 0.9 }], [2, { acc: 0.7 }]]),
        }),
        run_dir: runDir,
        num_metric_calls_by_discovery_of_new_program: 2,
      });

      expect(newIdx).toBe(1);
      expect(existsSync(`${runDir}/gepa_state.json`)).toBe(true);
      expect(existsSync(`${runDir}/candidates.json`)).toBe(true);
      expect(existsSync(`${runDir}/run_log.json`)).toBe(true);
      expect(JSON.parse(readFileSync(`${runDir}/generated_best_outputs_valset/task_1/iter_1_prog_1.json`, "utf8"))).toEqual({
        out: "better",
      });

      const loaded = GEPAState.load<unknown, number>(runDir);
      expect(loaded.is_consistent()).toBe(true);
      expect(loaded.program_candidates).toEqual(state.program_candidates);
      expect(loaded.prog_candidate_val_subscores[1]).toEqual(new Map([[1, 0.9], [2, 0.6]]));
      expect(loaded.program_at_pareto_front_valset.get(1)).toEqual(new Set([1]));
      expect(loaded.program_at_pareto_front_valset.get(2)).toEqual(new Set([0, 1]));
      expect(loaded.best_outputs_valset?.get(1)).toEqual([[1, { out: "better" }]]);
      expect(loaded.best_example_evals.get(1)?.map((entry) => entry.score)).toEqual([0.9, 0.4]);
      expect(loaded.total_num_evals).toBe(state.total_num_evals);
      expect(loaded.full_program_trace).toEqual([{ i: 0, note: "kept" }]);
    } finally {
      rmSync(runDir, { recursive: true, force: true });
    }
  });

  test("initialize_gepa_state loads existing run_dir state", () => {
    const runDir = mkdtempSync(`${tmpdir()}/gepa-ts-state-load-`);
    const logs: string[] = [];
    try {
      const original = initialize_gepa_state({
        run_dir: runDir,
        logger: { log: () => {} },
        seed_candidate: { prompt: "x" },
        seed_valset_evaluation: new ValsetEvaluation({
          scores_by_val_id: new Map([[0, 0.1]]),
          outputs_by_val_id: new Map([[0, "old"]]),
        }),
      });
      original.total_num_evals = 17;
      original.save(runDir);

      const loaded = initialize_gepa_state({
        run_dir: runDir,
        logger: { log: (message) => logs.push(message) },
        seed_candidate: { prompt: "ignored" },
        seed_valset_evaluation: new ValsetEvaluation({
          scores_by_val_id: new Map([[0, 1]]),
          outputs_by_val_id: new Map([[0, "new"]]),
        }),
      });

      expect(loaded.total_num_evals).toBe(17);
      expect(loaded.program_candidates[0]).toEqual({ prompt: "x" });
      expect(logs).toEqual(["Loading gepa state from run dir"]);
    } finally {
      rmSync(runDir, { recursive: true, force: true });
    }
  });
});
