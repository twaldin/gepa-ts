import { describe, expect, test } from "vitest";
import {
  SeededRandom,
  find_dominator_programs,
  idxmax,
  is_dominated,
  json_default,
  remove_dominated_programs,
  select_program_candidate_from_pareto_front,
} from "../../src/utils";
import {
  idxmax as public_idxmax,
  select_program_candidate_from_pareto_front as public_select_program_candidate_from_pareto_front,
} from "../../src/index.js";

describe("idxmax", () => {
  test("returns index of max", () => {
    expect(idxmax([0.1, 0.5, 0.3])).toBe(1);
  });

  test("returns first match on ties", () => {
    expect(idxmax([0.5, 0.5])).toBe(0);
  });
});

describe("json_default", () => {
  test("converts mapping-like and set-like values to JSON-friendly shapes", () => {
    expect(json_default(new Map([["a", 1]]))).toEqual({ a: 1 });
    expect(json_default(new Set([1, 2]))).toEqual([1, 2]);
    expect(json_default({ a: 1 })).toEqual({ a: 1 });
  });

  test("falls back to string rendering for primitives", () => {
    expect(json_default(42)).toBe("42");
    expect(json_default(null)).toBe("null");
  });
});

describe("public utility exports", () => {
  test("exports gepa_utils-compatible helpers from the package entrypoint", () => {
    const fronts = {
      a: new Set([0, 1]),
      b: new Set([0]),
    };

    expect(public_idxmax([0.1, 0.9, 0.3])).toBe(1);
    expect(public_select_program_candidate_from_pareto_front(fronts, [1, 0], new SeededRandom(1))).toBe(0);
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
  test("random() matches Python random.Random for integer seeds", () => {
    const rng = new SeededRandom(0);
    expect([rng.random(), rng.random(), rng.random()]).toEqual([
      0.8444218515250481,
      0.7579544029403025,
      0.420571580830845,
    ]);
  });

  test("randint matches Python random.Random inclusive bounds", () => {
    const rng = new SeededRandom(0);
    expect([rng.randint(0, 19), rng.randint(0, 19), rng.randint(0, 19)]).toEqual([12, 13, 1]);
  });

  test("randint inclusive bounds", () => {
    const rng = new SeededRandom(3);
    for (let i = 0; i < 20; i += 1) {
      const v = rng.randint(2, 4);
      expect(v >= 2 && v <= 4).toBe(true);
    }
  });

  test("shuffle matches Python random.Random", () => {
    const rng = new SeededRandom(0);
    const arr = Array.from({ length: 20 }, (_, i) => i);
    rng.shuffle(arr);
    expect(arr).toEqual([10, 18, 16, 14, 0, 17, 11, 2, 3, 9, 5, 7, 4, 19, 6, 15, 8, 1, 13, 12]);
  });

  test("sample matches Python random.Random pool path", () => {
    const rng = new SeededRandom(0);
    expect(rng.sample(Array.from({ length: 10 }, (_, i) => i), 2)).toEqual([6, 9]);
    expect(rng.sample(Array.from({ length: 10 }, (_, i) => i), 5)).toEqual([0, 4, 7, 3, 2]);
  });

  test("sample matches Python random.Random selected-set path", () => {
    const rng = new SeededRandom(0);
    expect(rng.sample(Array.from({ length: 100 }, (_, i) => i), 3)).toEqual([49, 97, 53]);
  });
});
