import { describe, expect, test } from "vitest";
import {
  ParetoCandidateSelector,
  CurrentBestCandidateSelector,
  EpsilonGreedyCandidateSelector,
  TopKParetoCandidateSelector,
} from "../../src/candidate_selector";
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
      program_full_scores_val_set: [0.1, 0.5, 0.3],
      per_program_tracked_scores: [0.1, 0.5, 0.3],
    };
    expect(selector.select_candidate_idx(state)).toBe(1);
  });

  test("matches upstream by selecting from full validation scores, not tracked scores", () => {
    const selector = new CurrentBestCandidateSelector();
    const state = {
      total_num_evals: 0,
      program_at_pareto_front_valset: new Map(),
      program_full_scores_val_set: [0.1, 0.9, 0.2],
      per_program_tracked_scores: [1.0, 0.1, 0.8],
    };
    expect(selector.select_candidate_idx(state)).toBe(1);
  });

  test("uses configured frontier mapping when state exposes non-instance fronts", () => {
    const state = {
      total_num_evals: 0,
      program_at_pareto_front_valset: new Map([[0, new Set([0])]]),
      get_pareto_front_mapping: () => new Map<unknown, Set<number>>([[["objective", "quality"], new Set([1])]]),
      per_program_tracked_scores: [0.1, 0.9],
    };
    const selector = new ParetoCandidateSelector(new SeededRandom(0));

    expect(selector.select_candidate_idx(state)).toBe(1);
  });
});

describe("EpsilonGreedyCandidateSelector", () => {
  test("epsilon zero always returns current best", () => {
    const selector = new EpsilonGreedyCandidateSelector(0, new SeededRandom(1));
    const state = {
      total_num_evals: 0,
      program_at_pareto_front_valset: new Map(),
      program_full_scores_val_set: [0.1, 0.5, 0.3],
      per_program_tracked_scores: [0.1, 0.5, 0.3],
    };
    expect(selector.select_candidate_idx(state)).toBe(1);
  });

  test("epsilon zero exploits by full validation scores, not tracked scores", () => {
    const selector = new EpsilonGreedyCandidateSelector(0, new SeededRandom(1));
    const state = {
      total_num_evals: 0,
      program_at_pareto_front_valset: new Map(),
      program_full_scores_val_set: [0.1, 0.9, 0.2],
      per_program_tracked_scores: [1.0, 0.1, 0.8],
    };
    expect(selector.select_candidate_idx(state)).toBe(1);
  });

  test("rejects epsilon outside [0, 1]", () => {
    expect(() => new EpsilonGreedyCandidateSelector(-0.1, new SeededRandom(1))).toThrow(/epsilon/);
    expect(() => new EpsilonGreedyCandidateSelector(1.1, new SeededRandom(1))).toThrow(/epsilon/);
  });
});

describe("TopKParetoCandidateSelector", () => {
  test("filters pareto fronts to the top k aggregate candidates", () => {
    const selector = new TopKParetoCandidateSelector(2, new SeededRandom(1));
    const state = {
      total_num_evals: 0,
      program_at_pareto_front_valset: new Map([
        [0, new Set([0])],
        [1, new Set([2])],
      ]),
      per_program_tracked_scores: [0.9, 0.8, 0.1],
    };

    expect(selector.select_candidate_idx(state)).not.toBe(2);
  });

  test("falls back to current best when no pareto front survives top k filtering", () => {
    const selector = new TopKParetoCandidateSelector(1, new SeededRandom(1));
    const state = {
      total_num_evals: 0,
      program_at_pareto_front_valset: new Map([[0, new Set([2])]]),
      per_program_tracked_scores: [0.9, 0.8, 0.1],
    };

    expect(selector.select_candidate_idx(state)).toBe(0);
  });
});
