import { describe, expect, it } from 'vitest';
import {
  MergeProposer,
  does_triplet_have_desirable_predictors,
  filter_ancestors,
  find_common_ancestor_pair,
  sample_and_attempt_merge_programs_by_common_predictors,
  type MergesPerformed,
} from '../../src/proposer/merge.js';
import { SeededRandom } from '../../src/utils.js';
import type { Candidate, DataLoader, LoggerProtocol } from '../../src/index.js';

const logger: LoggerProtocol = {
  log: () => undefined,
  log_metrics: () => undefined,
};

class StubValset implements DataLoader<number, { id: number }> {
  readonly length = 10;

  all_ids(): number[] {
    return Array.from({ length: this.length }, (_, idx) => idx);
  }

  fetch(ids: number[]): { id: number }[] {
    return ids.map((id) => ({ id }));
  }
}

type MergeTestState = {
  i: number;
  full_program_trace: Array<Record<string, unknown>>;
  program_full_scores_val_set: number[];
  per_program_tracked_scores: number[];
  program_candidates: Candidate[];
  parent_program_for_candidate: Array<Array<number | null>>;
  prog_candidate_val_subscores: Array<Map<number, number>>;
  total_num_evals: number;
  get_pareto_front_mapping(): Map<unknown, Set<number>>;
  cached_evaluate_full(
    candidate: Candidate,
    example_ids: number[],
    fetcher: (example_ids: number[]) => unknown,
    evaluator: (
      batch: unknown,
      candidate: Candidate,
      example_ids: number[],
    ) => [unknown[], number[], Record<string, number>[] | null] | Promise<[unknown[], number[], Record<string, number>[] | null]>,
  ): Promise<[Map<number, unknown>, Map<number, number>, Map<number, Record<string, number>> | null, number]>;
  increment_evals(count: number): void;
};

function make_state(prog_val_scores: Array<Map<number, number>>): MergeTestState {
  const state: MergeTestState = {
    i: 0,
    full_program_trace: [{}],
    program_full_scores_val_set: [0.1, 0.6, 0.7],
    per_program_tracked_scores: [0.1, 0.6, 0.7],
    program_candidates: [{ pred: 'base' }, { pred: 'base' }, { pred: 'p2' }],
    parent_program_for_candidate: [[null], [0], [0]],
    prog_candidate_val_subscores: prog_val_scores,
    total_num_evals: 0,
    get_pareto_front_mapping: () => new Map([[0, new Set([1])], [1, new Set([2])]]),
    cached_evaluate_full: async (candidate, example_ids, fetcher, evaluator) => {
      const [outputs, scores, objective_scores] = await evaluator(fetcher(example_ids), candidate, example_ids);
      return [
        new Map(example_ids.map((id, idx) => [id, outputs[idx]])),
        new Map(example_ids.map((id, idx) => [id, scores[idx] ?? 0])),
        objective_scores === null ? null : new Map(example_ids.map((id, idx) => [id, objective_scores[idx] ?? {}])),
        example_ids.length,
      ];
    },
    increment_evals(count) {
      this.total_num_evals += count;
    },
  };
  return state;
}

describe('proposer merge helpers', () => {
  it('detects desirable divergent predictors', () => {
    expect(does_triplet_have_desirable_predictors([{ pred: 'A' }, { pred: 'A' }, { pred: 'B' }], 0, 1, 2)).toBe(true);
    expect(does_triplet_have_desirable_predictors([{ pred: 'A' }, { pred: 'A' }, { pred: 'A' }], 0, 1, 2)).toBe(false);
  });

  it('filters common ancestors with upstream viability rules', () => {
    const candidates = [{ pred: 'A' }, { pred: 'A' }, { pred: 'B' }];
    expect(filter_ancestors(1, 2, [0], [[[1, 2, 0]], []], [0.1, 0.5, 0.6], candidates)).toEqual([]);
    expect(filter_ancestors(1, 2, [0], [[], []], [0.9, 0.5, 0.6], candidates)).toEqual([]);
    expect(filter_ancestors(1, 2, [0], [[], []], [0.1, 0.6, 0.7], candidates)).toEqual([0]);
  });

  it('finds common ancestor pairs with Python-compatible seeded sampling', () => {
    const rng = new SeededRandom(0);
    const result = find_common_ancestor_pair(
      rng,
      [[], [0], [0]],
      [1, 2],
      [[], []],
      [0.1, 0.6, 0.7],
      [{ pred: 'A' }, { pred: 'A' }, { pred: 'B' }],
      3,
    );
    expect(result).toEqual([1, 2, 0]);
  });

  it('creates combined programs and records merge descriptions', () => {
    const merges_performed: MergesPerformed = [[], []];
    const result = sample_and_attempt_merge_programs_by_common_predictors({
      agg_scores: [0.1, 0.6, 0.7],
      rng: new SeededRandom(0),
      merge_candidates: [1, 2],
      merges_performed,
      program_candidates: [{ pred: 'A' }, { pred: 'A' }, { pred: 'B' }],
      parent_program_for_candidate: [[], [0], [0]],
    });

    expect(result).toEqual([{ pred: 'B' }, 1, 2, 0]);
    expect(merges_performed[1]).toEqual([[1, 2, [2]]]);

    const second_attempt = sample_and_attempt_merge_programs_by_common_predictors({
      agg_scores: [0.1, 0.6, 0.7],
      rng: new SeededRandom(0),
      merge_candidates: [1, 2],
      merges_performed,
      program_candidates: [{ pred: 'A' }, { pred: 'A' }, { pred: 'B' }],
      parent_program_for_candidate: [[], [0], [0]],
    });
    expect(second_attempt).toBeNull();
  });

  it('respects validation-support overlap gates before recording descriptions', () => {
    const merges_performed: MergesPerformed = [[], []];
    const result = sample_and_attempt_merge_programs_by_common_predictors({
      agg_scores: [0.1, 0.6, 0.7],
      rng: new SeededRandom(0),
      merge_candidates: [1, 2],
      merges_performed,
      program_candidates: [{ pred: 'A' }, { pred: 'A' }, { pred: 'B' }],
      parent_program_for_candidate: [[], [0], [0]],
      has_val_support_overlap: () => false,
      max_attempts: 3,
    });
    expect(result).toBeNull();
    expect(merges_performed[1]).toEqual([]);
  });
});

describe('MergeProposer', () => {
  it('skips pairs below the validation overlap floor', async () => {
    const proposer = new MergeProposer({
      logger,
      valset: new StubValset(),
      evaluator: (batch) => [batch, batch.map(() => 0), null],
      use_merge: true,
      max_merge_invocations: 5,
      val_overlap_floor: 2,
      rng: new SeededRandom(0),
    });
    proposer.last_iter_found_new_program = true;
    proposer.merges_due = 1;

    const result = await proposer.propose(make_state([
      new Map([[0, 0.1], [1, 0.2]]),
      new Map([[0, 0.4], [1, 0.5]]),
      new Map([[1, 0.55]]),
    ]));

    expect(result).toBeNull();
    expect(proposer.merges_performed[0]).toEqual([]);
    expect(proposer.merges_performed[1]).toEqual([]);
  });

  it('returns a merge proposal for pairs meeting the overlap floor', async () => {
    const proposer = new MergeProposer({
      logger,
      valset: new StubValset(),
      evaluator: (batch) => [batch, batch.map(() => 0.9), null],
      use_merge: true,
      max_merge_invocations: 5,
      val_overlap_floor: 2,
      rng: new SeededRandom(0),
    });
    proposer.last_iter_found_new_program = true;
    proposer.merges_due = 1;

    const state = make_state([
      new Map([[0, 0.1], [1, 0.2]]),
      new Map([[0, 0.4], [1, 0.5]]),
      new Map([[0, 0.45], [1, 0.55]]),
    ]);
    const result = await proposer.propose(state);

    expect(result?.parent_program_ids).toEqual([1, 2]);
    expect(result?.subsample_indices).toEqual([1, 0, 1, 1, 1]);
    expect(proposer.merges_performed[0]).toEqual([[1, 2, 0]]);
    expect(proposer.merges_performed[1]).toEqual([[1, 2, [2]]]);
    expect(state.total_num_evals).toBe(5);
    expect(state.full_program_trace.at(-1)?.['merged']).toBe(true);
  });
});
