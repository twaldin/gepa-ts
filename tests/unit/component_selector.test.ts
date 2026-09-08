import { describe, expect, test } from "vitest";
import { AllReflectionComponentSelector, RoundRobinReflectionComponentSelector } from "../../src/component_selector";

describe("RoundRobinReflectionComponentSelector", () => {
  test("cycles predictor names and mutates pointer", () => {
    const selector = new RoundRobinReflectionComponentSelector();
    const state = {
      total_num_evals: 0,
      named_predictor_id_to_update_next_for_program_candidate: [0],
    };
    const candidate = { name_a: "a", name_b: "b" };

    expect(selector(state, [], [], 0, candidate)).toEqual(["name_a"]);
    expect(state.named_predictor_id_to_update_next_for_program_candidate[0]).toBe(1);

    expect(selector(state, [], [], 0, candidate)).toEqual(["name_b"]);
    expect(state.named_predictor_id_to_update_next_for_program_candidate[0]).toBe(0);

    expect(selector(state, [], [], 0, candidate)).toEqual(["name_a"]);
    expect(state.named_predictor_id_to_update_next_for_program_candidate[0]).toBe(1);
  });

  test("uses upstream state predictor order instead of candidate key order when available", () => {
    const selector = new RoundRobinReflectionComponentSelector();
    const state = {
      total_num_evals: 0,
      named_predictor_id_to_update_next_for_program_candidate: [0],
      list_of_named_predictors: ["second", "first"],
    };
    const candidate = { first: "a", second: "b" };

    expect(selector(state, [], [], 0, candidate)).toEqual(["second"]);
    expect(selector(state, [], [], 0, candidate)).toEqual(["first"]);
  });
});

describe("AllReflectionComponentSelector", () => {
  test("returns all candidate keys", () => {
    const selector = new AllReflectionComponentSelector();
    const candidate = { first: "a", second: "b" };
    expect(selector({ total_num_evals: 0 }, [], [], 0, candidate)).toEqual(["first", "second"]);
  });
});
