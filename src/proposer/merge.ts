import type {
  Candidate,
  CandidateProposal,
  DataId,
  DataInst,
  DataLoader,
  GEPACallback,
  RolloutOutput,
} from '../types.js';
import { notify_callbacks } from '../callbacks.js';
import type { ObjectiveScores } from '../state.js';
import { find_dominator_programs, SeededRandom } from '../utils.js';

export type AncestorLog = [number, number, number];
export type MergeDescription = [number, number, number[]];
export type MergeAttempt = [Candidate, number, number, number] | null;
export type MergesPerformed = [AncestorLog[], MergeDescription[]];
export type MergeLogger = { log(message: string): void };

export function does_triplet_have_desirable_predictors(
  program_candidates: Candidate[],
  ancestor: number,
  id1: number,
  id2: number,
): boolean {
  const pred_names = Object.keys(program_candidates[ancestor] ?? {});
  for (const pred_name of pred_names) {
    const pred_anc = program_candidates[ancestor]?.[pred_name];
    const pred_id1 = program_candidates[id1]?.[pred_name];
    const pred_id2 = program_candidates[id2]?.[pred_name];
    if ((pred_anc === pred_id1 || pred_anc === pred_id2) && pred_id1 !== pred_id2) {
      return true;
    }
  }
  return false;
}

function has_ancestor_log(logs: AncestorLog[], id1: number, id2: number, ancestor: number): boolean {
  return logs.some(([left, right, anc]) => left === id1 && right === id2 && anc === ancestor);
}

function has_merge_description(logs: MergeDescription[], id1: number, id2: number, desc: number[]): boolean {
  return logs.some(([left, right, existing]) =>
    left === id1 && right === id2 && existing.length === desc.length && existing.every((value, idx) => value === desc[idx])
  );
}

export function filter_ancestors(
  i: number,
  j: number,
  common_ancestors: Iterable<number>,
  merges_performed: MergesPerformed,
  agg_scores: number[],
  program_candidates: Candidate[],
): number[] {
  const filtered_ancestors: number[] = [];
  for (const ancestor of common_ancestors) {
    if (has_ancestor_log(merges_performed[0], i, j, ancestor)) {
      continue;
    }
    if ((agg_scores[ancestor] ?? 0) > (agg_scores[i] ?? 0) || (agg_scores[ancestor] ?? 0) > (agg_scores[j] ?? 0)) {
      continue;
    }
    if (!does_triplet_have_desirable_predictors(program_candidates, ancestor, i, j)) {
      continue;
    }
    filtered_ancestors.push(ancestor);
  }
  return filtered_ancestors;
}

function get_ancestors(parent_list: Array<Array<number | null>>, node: number, ancestors_found: Set<number>): number[] {
  const parents = parent_list[node] ?? [];
  for (const parent of parents) {
    if (parent !== null && !ancestors_found.has(parent)) {
      ancestors_found.add(parent);
      get_ancestors(parent_list, parent, ancestors_found);
    }
  }
  return [...ancestors_found];
}

export function find_common_ancestor_pair(
  rng: SeededRandom,
  parent_list: Array<Array<number | null>>,
  program_indexes: number[],
  merges_performed: MergesPerformed,
  agg_scores: number[],
  program_candidates: Candidate[],
  max_attempts = 10,
): [number, number, number] | null {
  for (let attempt = 0; attempt < max_attempts; attempt += 1) {
    if (program_indexes.length < 2) {
      return null;
    }
    let [i, j] = rng.sample(program_indexes, 2);
    if (i === undefined || j === undefined) {
      return null;
    }
    if (i === j) {
      continue;
    }
    if (j < i) {
      [i, j] = [j, i];
    }

    const ancestors_i = new Set(get_ancestors(parent_list, i, new Set()));
    const ancestors_j = new Set(get_ancestors(parent_list, j, new Set()));
    if (ancestors_i.has(j) || ancestors_j.has(i)) {
      continue;
    }

    const common_ancestors = [...ancestors_i].filter((ancestor) => ancestors_j.has(ancestor));
    const filtered = filter_ancestors(i, j, common_ancestors, merges_performed, agg_scores, program_candidates);
    if (filtered.length > 0) {
      const [common_ancestor] = rng.choices(filtered, {
        k: 1,
        weights: filtered.map((ancestor) => agg_scores[ancestor] ?? 0),
      });
      if (common_ancestor === undefined) {
        return null;
      }
      return [i, j, common_ancestor];
    }
  }
  return null;
}

function assert_same_predictors(program_candidates: Candidate[], ancestor: number, id1: number, id2: number): string[] {
  const ancestor_keys = Object.keys(program_candidates[ancestor] ?? {}).sort();
  const id1_keys = Object.keys(program_candidates[id1] ?? {}).sort();
  const id2_keys = Object.keys(program_candidates[id2] ?? {}).sort();
  const same = ancestor_keys.length === id1_keys.length &&
    ancestor_keys.length === id2_keys.length &&
    ancestor_keys.every((key, idx) => key === id1_keys[idx] && key === id2_keys[idx]);
  if (!same) {
    throw new Error('Predictors should be the same across all programs');
  }
  return ancestor_keys;
}

export function sample_and_attempt_merge_programs_by_common_predictors(opts: {
  agg_scores: number[];
  rng: SeededRandom;
  merge_candidates: number[];
  merges_performed: MergesPerformed;
  program_candidates: Candidate[];
  parent_program_for_candidate: Array<Array<number | null>>;
  has_val_support_overlap?: ((id1: number, id2: number) => boolean) | null;
  max_attempts?: number;
}): MergeAttempt {
  const {
    agg_scores,
    rng,
    merge_candidates,
    merges_performed,
    program_candidates,
    parent_program_for_candidate,
    has_val_support_overlap,
    max_attempts = 10,
  } = opts;

  if (merge_candidates.length < 2 || parent_program_for_candidate.length < 3) {
    return null;
  }

  for (let attempt = 0; attempt < max_attempts; attempt += 1) {
    const ids_to_merge = find_common_ancestor_pair(
      rng,
      parent_program_for_candidate,
      [...merge_candidates],
      merges_performed,
      agg_scores,
      program_candidates,
      max_attempts,
    );
    if (ids_to_merge === null) {
      continue;
    }
    const [id1, id2, ancestor] = ids_to_merge;
    if (has_ancestor_log(merges_performed[0], id1, id2, ancestor)) {
      continue;
    }
    if ((agg_scores[ancestor] ?? 0) > (agg_scores[id1] ?? 0) || (agg_scores[ancestor] ?? 0) > (agg_scores[id2] ?? 0)) {
      throw new Error('Ancestor should not be better than its descendants');
    }
    if (id1 === id2) {
      throw new Error('Cannot merge the same program');
    }

    const ancestor_program = program_candidates[ancestor] ?? {};
    const new_program: Candidate = { ...ancestor_program };
    const new_prog_desc: number[] = [];

    for (const pred_name of assert_same_predictors(program_candidates, ancestor, id1, id2)) {
      const pred_anc = program_candidates[ancestor]?.[pred_name];
      const pred_id1 = program_candidates[id1]?.[pred_name];
      const pred_id2 = program_candidates[id2]?.[pred_name];
      if ((pred_anc === pred_id1 || pred_anc === pred_id2) && pred_id1 !== pred_id2) {
        const new_value_idx = pred_anc === pred_id1 ? id2 : id1;
        const new_value = program_candidates[new_value_idx]?.[pred_name];
        if (new_value !== undefined) {
          new_program[pred_name] = new_value;
        }
        new_prog_desc.push(new_value_idx);
      } else if (pred_anc !== pred_id1 && pred_anc !== pred_id2) {
        const score1 = agg_scores[id1] ?? 0;
        const score2 = agg_scores[id2] ?? 0;
        const prog_to_get_instruction_from = score1 > score2 ? id1 : (score2 > score1 ? id2 : rng.choice([id1, id2]));
        const new_value = program_candidates[prog_to_get_instruction_from]?.[pred_name];
        if (new_value !== undefined) {
          new_program[pred_name] = new_value;
        }
        new_prog_desc.push(prog_to_get_instruction_from);
      } else if (pred_id1 === pred_id2) {
        const new_value = program_candidates[id1]?.[pred_name];
        if (new_value !== undefined) {
          new_program[pred_name] = new_value;
        }
        new_prog_desc.push(id1);
      } else {
        throw new Error('Unexpected case in predictor merging logic');
      }
    }

    if (has_merge_description(merges_performed[1], id1, id2, new_prog_desc)) {
      continue;
    }
    if (has_val_support_overlap && !has_val_support_overlap(id1, id2)) {
      continue;
    }

    merges_performed[1].push([id1, id2, new_prog_desc]);
    return [new_program, id1, id2, ancestor];
  }
  return null;
}

function map_to_record(map: Map<unknown, Set<number>>): Record<string | number, Set<number>> {
  const out: Record<string | number, Set<number>> = {};
  let idx = 0;
  for (const value of map.values()) {
    out[idx] = new Set(value);
    idx += 1;
  }
  return out;
}

type MergeState<TRolloutOutput, TDataId extends DataId> = {
  i: number;
  full_program_trace: Array<Record<string, unknown>>;
  program_full_scores_val_set: number[];
  per_program_tracked_scores?: number[];
  program_candidates: Candidate[];
  parent_program_for_candidate: Array<Array<number | null>>;
  prog_candidate_val_subscores: Array<Map<TDataId, number>>;
  get_pareto_front_mapping(): Map<unknown, Set<number>>;
  cached_evaluate_full(
    candidate: Candidate,
    example_ids: TDataId[],
    fetcher: (example_ids: TDataId[]) => unknown | Promise<unknown>,
    evaluator: (
      batch: unknown,
      candidate: Candidate,
      example_ids: TDataId[],
    ) => [TRolloutOutput[], number[], ObjectiveScores[] | null] | Promise<[TRolloutOutput[], number[], ObjectiveScores[] | null]>,
  ): Promise<[Map<TDataId, TRolloutOutput>, Map<TDataId, number>, Map<TDataId, ObjectiveScores> | null, number]>;
  increment_evals(count: number): void;
};

export class MergeProposer<TRolloutOutput = RolloutOutput, TDataId extends DataId = DataId, TDataInst = DataInst> {
  readonly logger: MergeLogger;
  readonly valset: DataLoader<TDataId, TDataInst>;
  readonly evaluator: (
    batch: TDataInst[],
    candidate: Candidate,
    example_ids: TDataId[],
  ) => [TRolloutOutput[], number[], ObjectiveScores[] | null] | Promise<[TRolloutOutput[], number[], ObjectiveScores[] | null]>;
  readonly use_merge: boolean;
  readonly max_merge_invocations: number;
  readonly rng: SeededRandom;
  readonly callbacks: GEPACallback[] | null;
  readonly val_overlap_floor: number;
  merges_due: number;
  total_merges_tested: number;
  merges_performed: MergesPerformed;
  last_iter_found_new_program: boolean;

  constructor(opts: {
    logger: MergeLogger;
    valset: DataLoader<TDataId, TDataInst>;
    evaluator: (
      batch: TDataInst[],
      candidate: Candidate,
      example_ids: TDataId[],
    ) => [TRolloutOutput[], number[], ObjectiveScores[] | null] | Promise<[TRolloutOutput[], number[], ObjectiveScores[] | null]>;
    use_merge: boolean;
    max_merge_invocations: number;
    val_overlap_floor?: number;
    rng?: SeededRandom | null;
    callbacks?: GEPACallback[] | null;
  }) {
    if ((opts.val_overlap_floor ?? 5) <= 0) {
      throw new Error('val_overlap_floor should be a positive integer');
    }
    this.logger = opts.logger;
    this.valset = opts.valset;
    this.evaluator = opts.evaluator;
    this.use_merge = opts.use_merge;
    this.max_merge_invocations = opts.max_merge_invocations;
    this.val_overlap_floor = opts.val_overlap_floor ?? 5;
    this.rng = opts.rng ?? new SeededRandom(0);
    this.callbacks = opts.callbacks ?? null;
    this.merges_due = 0;
    this.total_merges_tested = 0;
    this.merges_performed = [[], []];
    this.last_iter_found_new_program = false;
  }

  schedule_if_needed(): void {
    if (this.use_merge && this.total_merges_tested < this.max_merge_invocations) {
      this.merges_due += 1;
    }
  }

  select_eval_subsample_for_merged_program(
    scores1: Map<TDataId, number>,
    scores2: Map<TDataId, number>,
    num_subsample_ids = 5,
  ): TDataId[] {
    const common_ids = [...scores1.keys()].filter((idx) => scores2.has(idx));
    const p1 = common_ids.filter((idx) => (scores1.get(idx) ?? 0) > (scores2.get(idx) ?? 0));
    const p2 = common_ids.filter((idx) => (scores2.get(idx) ?? 0) > (scores1.get(idx) ?? 0));
    const p3 = common_ids.filter((idx) => !p1.includes(idx) && !p2.includes(idx));

    const n_each = Math.max(1, Math.ceil(num_subsample_ids / 3));
    const selected: TDataId[] = [];
    for (const bucket of [p1, p2, p3]) {
      if (selected.length >= num_subsample_ids) {
        break;
      }
      const available = bucket.filter((idx) => !selected.includes(idx));
      const take = Math.min(available.length, n_each, num_subsample_ids - selected.length);
      if (take > 0) {
        selected.push(...this.rng.sample(available, take));
      }
    }

    const remaining = num_subsample_ids - selected.length;
    if (remaining > 0) {
      const unused = common_ids.filter((idx) => !selected.includes(idx));
      if (unused.length >= remaining) {
        selected.push(...this.rng.sample(unused, remaining));
      } else if (common_ids.length > 0) {
        selected.push(...this.rng.choices(common_ids, { k: remaining }));
      }
    }

    return selected.slice(0, num_subsample_ids);
  }

  async propose(state: MergeState<TRolloutOutput, TDataId>): Promise<CandidateProposal<TDataId> | null> {
    const i = state.i + 1;
    if (state.full_program_trace.length === 0) {
      state.full_program_trace.push({});
    }
    state.full_program_trace[state.full_program_trace.length - 1]!['invoked_merge'] = true;

    if (!(this.use_merge && this.last_iter_found_new_program && this.merges_due > 0)) {
      this.logger.log(`Iteration ${i}: No merge candidates scheduled`);
      return null;
    }

    const tracked_scores = state.per_program_tracked_scores ?? state.program_full_scores_val_set;
    const merge_candidates = find_dominator_programs(map_to_record(state.get_pareto_front_mapping()), tracked_scores);

    const has_val_support_overlap = (id1: number, id2: number): boolean => {
      const scores1 = state.prog_candidate_val_subscores[id1] ?? new Map<TDataId, number>();
      const scores2 = state.prog_candidate_val_subscores[id2] ?? new Map<TDataId, number>();
      const common_count = [...scores1.keys()].filter((idx) => scores2.has(idx)).length;
      return common_count >= this.val_overlap_floor;
    };

    const merge_output = sample_and_attempt_merge_programs_by_common_predictors({
      agg_scores: [...tracked_scores],
      rng: this.rng,
      merge_candidates,
      merges_performed: this.merges_performed,
      program_candidates: state.program_candidates,
      parent_program_for_candidate: state.parent_program_for_candidate,
      has_val_support_overlap,
    });

    if (merge_output === null) {
      this.logger.log(`Iteration ${i}: No merge candidates found`);
      return null;
    }

    const [new_program, id1, id2, ancestor] = merge_output;
    const trace = state.full_program_trace[state.full_program_trace.length - 1]!;
    trace['merged'] = true;
    trace['merged_entities'] = [id1, id2, ancestor];
    this.merges_performed[0].push([id1, id2, ancestor]);
    this.logger.log(`Iteration ${i}: Merged programs ${id1} and ${id2} via ancestor ${ancestor}`);

    const scores1 = state.prog_candidate_val_subscores[id1] ?? new Map<TDataId, number>();
    const scores2 = state.prog_candidate_val_subscores[id2] ?? new Map<TDataId, number>();
    const subsample_ids = this.select_eval_subsample_for_merged_program(scores1, scores2);
    if (subsample_ids.length === 0) {
      this.logger.log(`Iteration ${i}: Skipping merge of ${id1} and ${id2} due to insufficient overlapping val coverage`);
      return null;
    }

    const id1_sub_scores = subsample_ids.map((idx) => {
      const score = scores1.get(idx);
      if (score === undefined) {
        throw new Error('Merged subsample id missing from first parent scores');
      }
      return score;
    });
    const id2_sub_scores = subsample_ids.map((idx) => {
      const score = scores2.get(idx);
      if (score === undefined) {
        throw new Error('Merged subsample id missing from second parent scores');
      }
      return score;
    });
    trace['subsample_ids'] = subsample_ids;

    const mini_devset = this.valset.fetch(subsample_ids);
    notify_callbacks(this.callbacks ?? undefined, 'on_evaluation_start', {
      iteration: i,
      candidate_idx: null,
      batch_size: mini_devset.length,
      capture_traces: false,
      parent_ids: [id1, id2],
      inputs: mini_devset,
      is_seed_candidate: false,
    });

    const [outputs_by_id, scores_by_id, objective_by_id, actual_evals_count] = await state.cached_evaluate_full(
      new_program,
      subsample_ids,
      (example_ids) => this.valset.fetch(example_ids),
      (batch, candidate, example_ids) => this.evaluator(batch as TDataInst[], candidate, example_ids),
    );
    const new_sub_scores = subsample_ids.map((eid) => {
      const score = scores_by_id.get(eid);
      if (score === undefined) {
        throw new Error(`Missing merge score for example id: ${String(eid)}`);
      }
      return score;
    });
    const outputs = subsample_ids.map((eid) => outputs_by_id.get(eid));

    notify_callbacks(this.callbacks ?? undefined, 'on_evaluation_end', {
      iteration: i,
      candidate_idx: null,
      scores: new_sub_scores,
      has_trajectories: false,
      parent_ids: [id1, id2],
      outputs,
      trajectories: null,
      objective_scores: objective_by_id === null ? null : subsample_ids.map((eid) => objective_by_id.get(eid) ?? {}),
      is_seed_candidate: false,
    });

    trace['id1_subsample_scores'] = id1_sub_scores;
    trace['id2_subsample_scores'] = id2_sub_scores;
    trace['new_program_subsample_scores'] = new_sub_scores;
    state.increment_evals(actual_evals_count);

    return {
      candidate: new_program,
      parent_program_ids: [id1, id2],
      subsample_indices: subsample_ids,
      subsample_scores_before: [id1_sub_scores.reduce((acc, value) => acc + value, 0), id2_sub_scores.reduce((acc, value) => acc + value, 0)],
      subsample_scores_after: new_sub_scores,
      tag: 'merge',
      metadata: { ancestor },
    };
  }
}
