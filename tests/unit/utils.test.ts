import { describe, expect, test } from "vitest";
import {
  SeededRandom,
  find_dominator_programs,
  idxmax,
  is_dominated,
  remove_dominated_programs,
  select_program_candidate_from_pareto_front,
} from "../../src/utils";

describe("idxmax", () => {
  test("returns index of max", () => {
    expect(idxmax([0.1, 0.5, 0.3])).toBe(1);
  });

  test("returns first match on ties", () => {
    expect(idxmax([0.5, 0.5])).toBe(0);
  });
});

describe("is_dominated", () => {
  const fronts = {
    a: new Set([0, 1]),
    b: new Set([0]),
  };

  test("true when every front containing y has a dominator", () => {
    expect(is_dominated(1, new Set([0]), fronts)).toBe(true);
  });

  test("false when at least one front lacks a dominator", () => {
    expect(is_dominated(1, new Set<number>(), fronts)).toBe(false);
  });
});

describe("remove_dominated_programs", () => {
  test("removes dominated candidate", () => {
    const fronts = {
      a: new Set([0, 1]),
      b: new Set([0]),
    };
    const out = remove_dominated_programs(fronts);
    expect(out.a).toEqual(new Set([0]));
    expect(out.b).toEqual(new Set([0]));
  });

  test("preserves non-dominated pair", () => {
    const fronts = {
      a: new Set([0]),
      b: new Set([1]),
    };
    const out = remove_dominated_programs(fronts);
    expect(out.a).toEqual(new Set([0]));
    expect(out.b).toEqual(new Set([1]));
  });
});

describe("find_dominator_programs", () => {
  test("returns unique dominators", () => {
    const fronts = {
      a: new Set([0, 1]),
      b: new Set([0]),
    };
    const out = find_dominator_programs(fronts, [1, 1]);
    expect(out).toEqual([0]);
  });
});

describe("select_program_candidate_from_pareto_front", () => {
  test("stable across same seed", () => {
    const fronts = {
      a: new Set([0, 1]),
      b: new Set([0]),
      c: new Set([0]),
    };
    const rng1 = new SeededRandom(0);
    const rng2 = new SeededRandom(0);
    const pick1 = select_program_candidate_from_pareto_front(fronts, [0.2, 0.1], rng1);
    const pick2 = select_program_candidate_from_pareto_front(fronts, [0.2, 0.1], rng2);
    expect(pick1).toBe(pick2);
  });

  test("selection comes from surviving front", () => {
    const fronts = {
      a: new Set([0, 1]),
      b: new Set([0]),
    };
    const pick = select_program_candidate_from_pareto_front(fronts, [1, 0], new SeededRandom(1));
    expect([0]).toContain(pick);
  });
});

describe("SeededRandom", () => {
  test("random() deterministic for same seed", () => {
    const a = new SeededRandom(0);
    const b = new SeededRandom(0);
    expect(a.random()).toBe(b.random());
  });

  test("randint inclusive bounds", () => {
    const rng = new SeededRandom(3);
    for (let i = 0; i < 20; i += 1) {
      const v = rng.randint(2, 4);
      expect(v >= 2 && v <= 4).toBe(true);
    }
  });

  test("shuffle preserves members", () => {
    const rng = new SeededRandom(2);
    const arr = [1, 2, 3, 4, 5];
    rng.shuffle(arr);
    expect([...arr].sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5]);
  });
});
