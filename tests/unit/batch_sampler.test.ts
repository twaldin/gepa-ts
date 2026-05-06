import { describe, expect, test } from "vitest";
import { EpochShuffledBatchSampler } from "../../src/batch_sampler";
import { SeededRandom } from "../../src/utils";

function number_loader(ids: number[]) {
  return { all_ids: () => [...ids], fetch: (fetch_ids: number[]) => fetch_ids, length: ids.length };
}

describe("EpochShuffledBatchSampler", () => {
  test("pads shuffled ids to a minibatch multiple", () => {
    const sampler = new EpochShuffledBatchSampler<number, number>(3, new SeededRandom(7));
    const loader = number_loader([0, 1, 2, 3, 4]);

    sampler.next_minibatch_ids(loader, { total_num_evals: 0, i: 0 });

    expect(sampler.shuffled_ids).toHaveLength(6);
    expect(sampler.shuffled_ids.length % sampler.minibatch_size).toBe(0);
  });

  test("uses padded id at epoch boundary instead of wrapping", () => {
    const sampler = new EpochShuffledBatchSampler<number, number>(3, new SeededRandom(7));
    const loader = number_loader([0, 1, 2, 3, 4]);
    const state = { total_num_evals: 0, i: 0 };

    const first = sampler.next_minibatch_ids(loader, state);
    const padded_id = sampler.shuffled_ids[5];
    state.i = 1;
    const second = sampler.next_minibatch_ids(loader, state);

    expect(first).toEqual(sampler.shuffled_ids.slice(0, 3));
    expect(second).toEqual(sampler.shuffled_ids.slice(3, 6));
    expect(second[2]).toBe(padded_id);
    expect(second[2]).not.toBe(sampler.shuffled_ids[0]);
  });

  test("throws on empty loader", () => {
    const sampler = new EpochShuffledBatchSampler<number, number>(3, new SeededRandom(1));
    const loader = number_loader([]);
    expect(() => sampler.next_minibatch_ids(loader, { total_num_evals: 0, i: 0 })).toThrow(
      "Cannot sample a minibatch from an empty loader.",
    );
  });

  test("refreshes when state crosses padded epoch boundary", () => {
    const sampler = new EpochShuffledBatchSampler<number, number>(3, new SeededRandom(5));
    const loader = number_loader([0, 1, 2, 3, 4]);
    const state = { total_num_evals: 0, i: 0 };

    sampler.next_minibatch_ids(loader, state);
    expect(sampler.epoch).toBe(0);
    const before = [...sampler.shuffled_ids];

    state.i = 1;
    sampler.next_minibatch_ids(loader, state);
    expect(sampler.epoch).toBe(0);

    state.i = 2;
    sampler.next_minibatch_ids(loader, state);

    expect(sampler.epoch).toBe(1);
    expect(sampler.shuffled_ids).not.toEqual(before);
    expect(sampler.shuffled_ids).toHaveLength(6);
  });

  test("refreshes when trainset size changes", () => {
    const sampler = new EpochShuffledBatchSampler<number, number>(3, new SeededRandom(13));
    const state = { total_num_evals: 0, i: 0 };
    const loader_a = number_loader([0, 1, 2, 3, 4]);
    const loader_b = number_loader([0, 1, 2, 3, 4, 5, 6]);

    sampler.next_minibatch_ids(loader_a, state);
    const before = [...sampler.shuffled_ids];
    sampler.next_minibatch_ids(loader_b, state);

    expect(sampler.last_trainset_size).toBe(7);
    expect(sampler.shuffled_ids).not.toEqual(before);
    expect(sampler.shuffled_ids.length % sampler.minibatch_size).toBe(0);
  });

  test("deterministic with same seed", () => {
    const loader = number_loader([0, 1, 2, 3, 4]);
    const state = { total_num_evals: 0, i: 0 };
    const s1 = new EpochShuffledBatchSampler<number, number>(3, new SeededRandom(9));
    const s2 = new EpochShuffledBatchSampler<number, number>(3, new SeededRandom(9));

    expect(s1.next_minibatch_ids(loader, state)).toEqual(s2.next_minibatch_ids(loader, state));
  });
});
