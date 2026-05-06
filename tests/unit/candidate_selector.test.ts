import { describe, expect, test } from "vitest";
import { ParetoCandidateSelector, CurrentBestCandidateSelector } from "../../src/candidate_selector";
import { SeededRandom } from "../../src/utils";

describe("ParetoCandidateSelector", () => {
  test("returns 0 with one program", () => {
    const selector = new ParetoCandidateSelector(new SeededRandom(1));
    const state = {
      total_num_evals: 0,
      program_at_pareto_front_valset: new Map([[0, new Set([0])]]),
      per_program_tracked_scores: [0.2],
    };
    expect(selector.select_candidate_idx(state)).toBe(0);
  });

  test("returns dominator when one program dominates all val ids", () => {
    const selector = new ParetoCandidateSelector(new SeededRandom(3));
    const state = {
      total_num_evals: 0,
      program_at_pareto_front_valset: new Map([
        [0, new Set([0])],
        [1, new Set([0])],
      ]),
      per_program_tracked_scores: [0.9, 0.1],
    };
    expect(selector.select_candidate_idx(state)).toBe(0);
  });

  test("two consecutive calls are deterministic when only one candidate remains after dominance filtering", () => {
    const state = {
      total_num_evals: 0,
      program_at_pareto_front_valset: new Map([
        [0, new Set([0])],
        [1, new Set([0])],
      ]),
      per_program_tracked_scores: [0.9, 0.5],
    };
    const selector = new ParetoCandidateSelector(new SeededRandom(42));
    const pick1 = selector.select_candidate_idx(state);
    const pick2 = selector.select_candidate_idx(state);
    expect(pick1).toBe(pick2);
  });
});

describe("CurrentBestCandidateSelector", () => {
  test("returns argmax index", () => {
    const selector = new CurrentBestCandidateSelector();
    const state = {
      total_num_evals: 0,
      program_at_pareto_front_valset: new Map(),
      per_program_tracked_scores: [0.1, 0.5, 0.3],
    };
    expect(selector.select_candidate_idx(state)).toBe(1);
  });
});
