import { describe, expect, test } from "vitest";
import { OptimizeAnythingAdapter } from "../../src/adapter";
import { type Candidate, type EvaluatorOptState, type SideInfo } from "../../src/types";

describe("OptimizeAnythingAdapter", () => {
  test("accepts upstream-shaped callable constructor and stores compatibility options", async () => {
    const evaluator = async (_candidate: Candidate, example?: unknown) =>
      [example === "hit" ? 1 : 0, "output", { scores: { acc: 1 } }] as [number, unknown, SideInfo];
    const reflection_lm = async () => "new text";
    const adapter = new OptimizeAnythingAdapter(
      evaluator,
      reflection_lm,
      "{candidate}",
      false,
      3,
      null,
      2,
      "maximize accuracy",
      "background",
      "off",
      "/tmp/gepa-cache",
    );

    const result = await adapter.evaluate(["hit"], { prompt: "x" }, false);

    expect(adapter.reflection_lm).toBe(reflection_lm);
    expect(adapter.reflection_prompt_template).toBe("{candidate}");
    expect(adapter.parallel).toBe(false);
    expect(adapter.max_workers).toBe(3);
    expect(adapter.best_example_evals_k).toBe(2);
    expect(adapter.objective).toBe("maximize accuracy");
    expect(adapter.background).toBe("background");
    expect(adapter.cache_mode).toBe("off");
    expect(adapter.cache_dir).toBe("/tmp/gepa-cache");
    expect(result.scores).toEqual([1]);
    expect(result.outputs[0]).toEqual([1, { prompt: "x" }, { scores: { acc: 1 } }]);
  });

  test("passes per-example best evaluation history to direct evaluator calls", async () => {
    const seen_best_eval_counts: number[] = [];
    const evaluator = async (_candidate: Candidate, _example?: unknown, opt_state?: EvaluatorOptState) => {
      seen_best_eval_counts.push(opt_state?.best_example_evals.length ?? -1);
      const score = seen_best_eval_counts.length === 1 ? 0.2 : 0.8;
      return [score, undefined, { scores: { score } }] as [number, unknown, SideInfo];
    };
    const adapter = new OptimizeAnythingAdapter({
      evaluator,
      best_example_evals_k: 1,
      cache_mode: "off",
    });

    await adapter.evaluate(["same-example"], { prompt: "first" }, false);
    await adapter.evaluate(["same-example"], { prompt: "second" }, false);

    expect(seen_best_eval_counts).toEqual([0, 1]);
    expect(adapter._get_best_example_evals("same-example")).toEqual([
      { score: 0.8, side_info: { scores: { score: 0.8 } } },
    ]);
  });

  test("memory cache reuses evaluator results for matching candidate and example", async () => {
    let calls = 0;
    const evaluator = async () => {
      calls += 1;
      return [0.5, undefined, { calls }] as [number, unknown, SideInfo];
    };
    const adapter = new OptimizeAnythingAdapter({ evaluator, cache_mode: "memory" });
    const candidate = { prompt: "x" };

    const first = await adapter.evaluate(["example"], candidate, false);
    const second = await adapter.evaluate(["example"], candidate, false);

    expect(calls).toBe(1);
    expect(first.side_infos?.[0]).toEqual({ calls: 1 });
    expect(second.side_infos?.[0]).toEqual({ calls: 1 });
  });

  test("exposes upstream-compatible cache helper methods", async () => {
    const evaluator = async () => [0, undefined, {}] as [number, unknown, SideInfo];
    const adapter = new OptimizeAnythingAdapter({ evaluator });
    const key = adapter._cache_key({ b: "two", a: "one" }, { z: 2, a: [1, { b: true }] });

    expect(adapter._candidate_hash({ b: "two", a: "one" })).toBe("a5cc796d8f26a92a");
    expect(adapter._example_hash({ z: 2, a: [1, { b: true }] })).toBe("66fc4b55a894fe9d");
    expect(key).toEqual(["a5cc796d8f26a92a", "66fc4b55a894fe9d"]);
    expect(adapter._cache_filename(key)).toBe("a5cc796d8f26a92a_66fc4b55a894fe9d.pkl");
  });

  test("formats refinement attempt feedback like upstream JSON", async () => {
    const evaluator = async () => [0, undefined, {}] as [number, unknown, SideInfo];
    const adapter = new OptimizeAnythingAdapter({ evaluator });

    expect(adapter._format_all_attempts_feedback([
      { iteration: 0, candidate: { prompt: "x" }, score: 0.25, side_info: { scores: { acc: 0.25 } } },
    ])).toBe(`[
  {
    "iteration": 0,
    "candidate": {
      "prompt": "x"
    },
    "score": 0.25,
    "side_info": {
      "scores": {
        "acc": 0.25
      }
    }
  }
]`);
  });

  test("evaluate builds outputs/scores/trajectories and metric count", async () => {
    const side_info_0: SideInfo = { a: 1 };
    const side_info_1: SideInfo = { b: 2 };
    const evaluator = {
      call: async (_candidate: Candidate, example?: unknown) => {
        if (example === "example1") return [0.5, undefined, side_info_0] as [number, unknown, SideInfo];
        return [0.7, undefined, side_info_1] as [number, unknown, SideInfo];
      },
    };
    const adapter = new OptimizeAnythingAdapter({ evaluator });
    const candidate = { prompt: "x" };

    const result = await adapter.evaluate(["example1", "example2"], candidate, false);

    expect(result.scores).toEqual([0.5, 0.7]);
    expect(result.outputs[0]).toEqual([0.5, candidate, side_info_0]);
    expect(result.trajectories).toEqual([side_info_0, side_info_1]);
    expect(result.num_metric_calls).toBe(2);
  });

  test("evaluate empty batch", async () => {
    const evaluator = {
      call: async () => [1, undefined, {}] as [number, unknown, SideInfo],
    };
    const adapter = new OptimizeAnythingAdapter({ evaluator });

    const result = await adapter.evaluate([], { prompt: "x" }, false);

    expect(result.outputs).toEqual([]);
    expect(result.scores).toEqual([]);
    expect(result.trajectories).toEqual([]);
    expect(result.objective_scores).toEqual([]);
    expect(result.num_metric_calls).toBe(0);
  });

  test("objective_scores from side_info.scores", async () => {
    const evaluator = {
      call: async () => [0.5, undefined, { scores: { acc: 0.5 } }] as [number, unknown, SideInfo],
    };
    const adapter = new OptimizeAnythingAdapter({ evaluator });

    const result = await adapter.evaluate([1], { prompt: "x" }, false);

    expect(result.objective_scores?.[0]).toEqual({ acc: 0.5 });
  });

  test("objective_scores merge with component-specific scores", async () => {
    const evaluator = {
      call: async () =>
        [0.6, undefined, { scores: { global: 1 }, prompt_specific_info: { scores: { local: 2 } } }] as [
          number,
          unknown,
          SideInfo,
        ],
    };
    const adapter = new OptimizeAnythingAdapter({ evaluator });

    const result = await adapter.evaluate([1], { prompt: "x" }, false);

    expect(result.objective_scores?.[0]).toEqual({ global: 1, "prompt::local": 2 });
  });

  test("refiner evaluates refined candidate and keeps the best score", async () => {
    const seen_candidates: Candidate[] = [];
    const evaluator = {
      call: async (candidate: Candidate) => {
        seen_candidates.push({ ...candidate });
        const score = candidate.number === "42" ? 1 : 0;
        return [score, undefined, { scores: { accuracy: score } }] as [number, unknown, SideInfo];
      },
    };
    const adapter = new OptimizeAnythingAdapter({
      evaluator,
      refiner_config: {
        max_refinements: 1,
        refiner_lm: async () => '```json\n{"number":"42"}\n```',
      },
    });

    const result = await adapter.evaluate([{}], {
      number: "10",
      refiner_prompt: "Improve the number.",
    }, true);

    expect(seen_candidates).toEqual([
      { number: "10", refiner_prompt: "Improve the number." },
      { number: "42", refiner_prompt: "Improve the number." },
    ]);
    expect(result.scores).toEqual([1]);
    expect(result.num_metric_calls).toBe(2);
    expect(result.objective_scores?.[0]).toEqual({ accuracy: 1, "refiner_prompt::accuracy": 1 });
    const side_info = result.side_infos?.[0] as SideInfo;
    expect(side_info.scores).toEqual({ accuracy: 1 });
    expect(side_info.refiner_prompt_specific_info).toMatchObject({
      scores: { accuracy: 1 },
      Attempts: [
        { iteration: 0, score: 0 },
        { iteration: 1, score: 1 },
      ],
    });
  });

  test("refiner parse failures preserve original score and objective scores", async () => {
    const evaluator = {
      call: async () => [0.4, undefined, { scores: { accuracy: 0.4 } }] as [number, unknown, SideInfo],
    };
    const adapter = new OptimizeAnythingAdapter({
      evaluator,
      refiner_config: {
        max_refinements: 3,
        refiner_lm: async () => "not json",
      },
    });

    const result = await adapter.evaluate([{}], {
      number: "50",
      refiner_prompt: "Improve the number.",
    }, true);

    expect(result.scores).toEqual([0.4]);
    expect(result.num_metric_calls).toBe(1);
    expect(result.objective_scores?.[0]).toEqual({ accuracy: 0.4, "refiner_prompt::accuracy": 0.4 });
    const side_info = result.side_infos?.[0] as SideInfo;
    const refiner_info = side_info.refiner_prompt_specific_info as Record<string, unknown>;
    expect(refiner_info.scores).toEqual({ accuracy: 0.4 });
    expect(refiner_info.Attempts).toMatchObject([
      { iteration: 0, score: 0.4 },
      { iteration: 1, score: -1000000000 },
    ]);
  });

  test("refiner preserves unchanged fields, accepts equal scores, and continues refinements", async () => {
    const seen_candidates: Candidate[] = [];
    const refinements = [
      '```json\n{"number":"10"}\n```',
      '```json\n{"suffix":"kept"}\n```',
    ];
    const evaluator = {
      call: async (candidate: Candidate) => {
        seen_candidates.push({ ...candidate });
        return [1, undefined, { scores: { accuracy: 1 }, label: `${candidate.prefix}-${candidate.number}-${candidate.suffix}` }] as [
          number,
          unknown,
          SideInfo,
        ];
      },
    };
    const adapter = new OptimizeAnythingAdapter({
      evaluator,
      refiner_config: {
        max_refinements: 2,
        refiner_lm: async () => refinements.shift() ?? "{}",
      },
    });

    const result = await adapter.evaluate([{}], {
      prefix: "keep",
      number: "10",
      suffix: "base",
      refiner_prompt: "Keep all fields unless changing one.",
    }, true);

    expect(seen_candidates).toEqual([
      { prefix: "keep", number: "10", suffix: "base", refiner_prompt: "Keep all fields unless changing one." },
      { prefix: "keep", number: "10", suffix: "base", refiner_prompt: "Keep all fields unless changing one." },
      { prefix: "keep", number: "10", suffix: "kept", refiner_prompt: "Keep all fields unless changing one." },
    ]);
    expect(result.num_metric_calls).toBe(3);
    expect(result.outputs[0]?.[1]).toEqual({
      prefix: "keep",
      number: "10",
      suffix: "kept",
      refiner_prompt: "Keep all fields unless changing one.",
    });
    expect(result.side_infos?.[0]).toMatchObject({
      scores: { accuracy: 1 },
      label: "keep-10-kept",
    });
  });

  test("make_reflective_dataset filters and flattens component-specific info", () => {
    const evaluator = {
      call: async () => [0, undefined, {}] as [number, unknown, SideInfo],
    };
    const adapter = new OptimizeAnythingAdapter({ evaluator });

    const eval_batch = {
      outputs: [],
      scores: [0.1],
      trajectories: [
        {
          Inputs: "a",
          Outputs: "b",
          scores: { x: 1 },
          prompt_specific_info: { extra: "z" },
          other_specific_info: { ignored: "y" },
        },
      ],
    };

    const dataset = adapter.make_reflective_dataset({ prompt: "x" }, eval_batch, ["prompt"]);
    const record = dataset.prompt[0];

    expect(record.Inputs).toBe("a");
    expect(record.Outputs).toBe("b");
    expect(record["Scores (Higher is Better)"]).toEqual({ x: 1 });
    expect(record.extra).toBe("z");
    expect(record.other_specific_info).toBeUndefined();
    expect((record as Record<string, unknown>).ignored).toBeUndefined();
  });

  test("make_reflective_dataset handles multiple components and examples", () => {
    const evaluator = {
      call: async () => [0, undefined, {}] as [number, unknown, SideInfo],
    };
    const adapter = new OptimizeAnythingAdapter({ evaluator });

    const eval_batch = {
      outputs: [],
      scores: [0.1, 0.2],
      trajectories: [
        { shared: "one", prompt_specific_info: { p: 1 }, title_specific_info: { t: "x" } },
        { shared: "two", prompt_specific_info: { p: 2 }, title_specific_info: { t: "y" } },
      ],
    };

    const dataset = adapter.make_reflective_dataset({ prompt: "x", title: "z" }, eval_batch, ["prompt", "title"]);

    expect(dataset.prompt).toHaveLength(2);
    expect(dataset.title).toHaveLength(2);
    expect(dataset.prompt[0].p).toBe(1);
    expect(dataset.prompt[0].t).toBeUndefined();
    expect(dataset.title[0].t).toBe("x");
    expect(dataset.title[0].p).toBeUndefined();
  });
});
