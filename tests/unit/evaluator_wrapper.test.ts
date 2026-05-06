import { describe, expect, test } from "vitest";
import { EvaluatorWrapper } from "../../src/evaluator_wrapper";
import { STR_CANDIDATE_KEY, type Candidate, type SideInfo } from "../../src/types";

describe("EvaluatorWrapper", () => {
  test("single_instance + str_candidate + tuple return", async () => {
    const evaluator = (candidate: string | Candidate): [number, SideInfo] => {
      expect(candidate).toBe("x");
      return [0.5, { foo: "bar" }];
    };
    const wrapper = new EvaluatorWrapper(evaluator, true, true);

    await expect(wrapper.call({ [STR_CANDIDATE_KEY]: "x" })).resolves.toEqual([0.5, undefined, { foo: "bar" }]);
  });

  test("single_instance + str_candidate + number return", async () => {
    const evaluator = (candidate: string | Candidate): number => {
      expect(candidate).toBe("x");
      return 0.7;
    };
    const wrapper = new EvaluatorWrapper(evaluator, true, true);

    await expect(wrapper.call({ [STR_CANDIDATE_KEY]: "x" })).resolves.toEqual([0.7, undefined, {}]);
  });

  test("multi-task + dict candidate + tuple", async () => {
    const candidate = { prompt: "x" };
    const evaluator = (arg_candidate: string | Candidate, ctx: { example: unknown }): [number, SideInfo] => {
      expect(arg_candidate).toEqual(candidate);
      expect(ctx).toEqual({ example: 42 });
      return [0.3, {}];
    };
    const wrapper = new EvaluatorWrapper(evaluator, false, false);

    await expect(wrapper.call(candidate, 42)).resolves.toEqual([0.3, undefined, {}]);
  });

  test("exception + raise_on_exception=true rethrows", async () => {
    const wrapper = new EvaluatorWrapper(() => {
      throw new Error("boom");
    }, true, false, true);

    await expect(wrapper.call({ prompt: "x" })).rejects.toThrow("boom");
  });

  test("exception + raise_on_exception=false returns error side_info", async () => {
    const wrapper = new EvaluatorWrapper(
      () => {
        throw new Error("boom");
      },
      true,
      false,
      false,
    );

    const result = await wrapper.call({ prompt: "x" });
    expect(result[0]).toBe(0.0);
    expect(result[1]).toBeUndefined();
    expect(String(result[2].error)).toContain("boom");
  });

  test("awaits async evaluator", async () => {
    const wrapper = new EvaluatorWrapper(async () => {
      return [0.9, { ok: true }];
    }, true);

    await expect(wrapper.call({ prompt: "x" })).resolves.toEqual([0.9, undefined, { ok: true }]);
  });

  test("str_candidate_mode + missing STR_CANDIDATE_KEY uses empty string", async () => {
    const evaluator = (candidate: string | Candidate): number => {
      expect(candidate).toBe("");
      return 0.2;
    };
    const wrapper = new EvaluatorWrapper(evaluator, true, true);

    await expect(wrapper.call({ prompt: "x" })).resolves.toEqual([0.2, undefined, {}]);
  });
});
