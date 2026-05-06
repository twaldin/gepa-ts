import { describe, expect, test } from "vitest";
import { EpochShuffledBatchSampler } from "../../src/batch_sampler";
import { SeededRandom } from "../../src/utils";

describe("EpochShuffledBatchSampler", () => {
  test("returns minibatches and pads with least-frequent id quirk", () => {
    const sampler = new EpochShuffledBatchSampler<number, number>(3, new SeededRandom(7));
    const loader = { all_ids: () => [0, 1, 2, 3, 4], fetch: (ids: number[]) => ids, length: 5 };
    const state = { total_num_evals: 0, i: 0 };

    const first = sampler.next_minibatch_ids(loader, state);
    state.i = 1;
    const second = sampler.next_minibatch_ids(loader, state);

    expect(first).toHaveLength(3);
    expect(second).toHaveLength(3);
    expect(second[2]).toBe(sampler.shuffled_ids[0]);
  });

  test("throws on empty loader", () => {
    const sampler = new EpochShuffledBatchSampler<number, number>(3, new SeededRandom(1));
    const loader = { all_ids: () => [] as number[], fetch: (ids: number[]) => ids, length: 0 };
    expect(() => sampler.next_minibatch_ids(loader, { total_num_evals: 0, i: 0 })).toThrow();
  });

  test("reshuffles when crossing epoch boundary", () => {
    const sampler = new EpochShuffledBatchSampler<number, number>(3, new SeededRandom(5));
    const loader = { all_ids: () => [0, 1, 2, 3, 4], fetch: (ids: number[]) => ids, length: 5 };
    const state = { total_num_evals: 0, i: 0 };

    sampler.next_minibatch_ids(loader, state);
    const before = [...sampler.shuffled_ids];
    state.i = 2;
    sampler.next_minibatch_ids(loader, state);
    const after = [...sampler.shuffled_ids];

    expect(after).not.toEqual(before);
  });

  test("deterministic with same seed", () => {
    const loader = { all_ids: () => [0, 1, 2, 3, 4], fetch: (ids: number[]) => ids, length: 5 };
    const state = { total_num_evals: 0, i: 0 };
    const s1 = new EpochShuffledBatchSampler<number, number>(3, new SeededRandom(9));
    const s2 = new EpochShuffledBatchSampler<number, number>(3, new SeededRandom(9));

    expect(s1.next_minibatch_ids(loader, state)).toEqual(s2.next_minibatch_ids(loader, state));
  });
});
