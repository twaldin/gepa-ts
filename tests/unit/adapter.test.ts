import { describe, expect, test } from "vitest";
import { OptimizeAnythingAdapter } from "../../src/adapter";
import { type Candidate, type SideInfo } from "../../src/types";

describe("OptimizeAnythingAdapter", () => {
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
