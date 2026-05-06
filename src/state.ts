import type { Candidate } from "./types";

export type ObjectiveScores = Record<string, number>;
export type SupportedFrontierType = "instance" | "objective" | "hybrid" | "cartesian";

export class ValsetEvaluation<RolloutOutput = unknown, DataId extends string | number = string | number> {
  outputs_by_val_id: Map<DataId, RolloutOutput>;
  scores_by_val_id: Map<DataId, number>;
  objective_scores_by_val_id: Map<DataId, ObjectiveScores> | null;

  constructor(opts: {
    outputs_by_val_id: Map<DataId, RolloutOutput>;
    scores_by_val_id: Map<DataId, number>;
    objective_scores_by_val_id?: Map<DataId, ObjectiveScores> | null;
  }) {
    this.outputs_by_val_id = opts.outputs_by_val_id;
    this.scores_by_val_id = opts.scores_by_val_id;
    this.objective_scores_by_val_id = opts.objective_scores_by_val_id ?? null;
  }
}

export class GEPAState<RolloutOutput = unknown, DataId extends string | number = string | number> {
  program_candidates: Candidate[];
  prog_candidate_val_subscores: Map<DataId, number>[];
  prog_candidate_objective_scores: ObjectiveScores[];
  parent_program_for_candidate: (number | null)[][];
  pareto_front_valset: Map<DataId, number>;
  program_at_pareto_front_valset: Map<DataId, Set<number>>;
  objective_pareto_front: ObjectiveScores;
  program_at_pareto_front_objectives: Map<string, Set<number>>;
  list_of_named_predictors: string[];
  named_predictor_id_to_update_next_for_program_candidate: number[];
  i: number;
  num_full_ds_evals: number;
  total_num_evals: number;
  num_metric_calls_by_discovery: number[];
  full_program_trace: Record<string, unknown>[];
  best_outputs_valset: Map<DataId, [number, RolloutOutput][]> | null;
  frontier_type: SupportedFrontierType;
  evaluation_cache: null;
  adapter_state: Record<string, unknown>;

  private _budget_hooks?: Array<(new_total: number, delta: number) => void>;

  constructor(opts: {
    seed_candidate: Candidate;
    base_evaluation: ValsetEvaluation<RolloutOutput, DataId>;
    track_best_outputs?: boolean;
    frontier_type?: SupportedFrontierType;
  }) {
    this.frontier_type = opts.frontier_type ?? "instance";
    if (this.frontier_type !== "instance") {
      throw new Error(`frontier_type "${this.frontier_type}" not supported in v1`);
    }

    this.program_candidates = [{ ...opts.seed_candidate }];
    this.prog_candidate_val_subscores = [new Map(opts.base_evaluation.scores_by_val_id)];

    const base_objective_aggregates = GEPAState._aggregate_objective_scores(opts.base_evaluation.objective_scores_by_val_id);
    this.prog_candidate_objective_scores = [base_objective_aggregates];

    this.parent_program_for_candidate = [[null]];
    this.pareto_front_valset = new Map(opts.base_evaluation.scores_by_val_id);
    this.program_at_pareto_front_valset = new Map();
    for (const val_id of opts.base_evaluation.scores_by_val_id.keys()) {
      this.program_at_pareto_front_valset.set(val_id, new Set([0]));
    }
    this.objective_pareto_front = { ...base_objective_aggregates };
    this.program_at_pareto_front_objectives = new Map();
    for (const objective of Object.keys(base_objective_aggregates)) {
      this.program_at_pareto_front_objectives.set(objective, new Set([0]));
    }

    this.list_of_named_predictors = Object.keys(opts.seed_candidate);
    this.named_predictor_id_to_update_next_for_program_candidate = [0];
    this.i = -1;
    this.num_full_ds_evals = 0;
    this.total_num_evals = 0;
    this.num_metric_calls_by_discovery = [0];
    this.full_program_trace = [];
    this.best_outputs_valset = opts.track_best_outputs
      ? new Map(Array.from(opts.base_evaluation.outputs_by_val_id.entries()).map(([val_id, output]) => [val_id, [[0, output]]]))
      : null;
    this.evaluation_cache = null;
    this.adapter_state = {};
  }

  is_consistent(): boolean {
    const n = this.program_candidates.length;
    if (n !== this.parent_program_for_candidate.length) {
      throw new Error("inconsistent state: parent_program_for_candidate length mismatch");
    }
    if (n !== this.named_predictor_id_to_update_next_for_program_candidate.length) {
      throw new Error("inconsistent state: named predictor pointer length mismatch");
    }
    if (n !== this.prog_candidate_val_subscores.length) {
      throw new Error("inconsistent state: val subscores length mismatch");
    }
    if (n !== this.prog_candidate_objective_scores.length) {
      throw new Error("inconsistent state: objective scores length mismatch");
    }
    if (n !== this.num_metric_calls_by_discovery.length) {
      throw new Error("inconsistent state: metric calls list length mismatch");
    }
    if (this.pareto_front_valset.size !== this.program_at_pareto_front_valset.size) {
      throw new Error("inconsistent state: valset front mapping size mismatch");
    }

    const pareto_ids = new Set(this.pareto_front_valset.keys());
    const mapping_ids = new Set(this.program_at_pareto_front_valset.keys());
    if (pareto_ids.size !== mapping_ids.size) {
      throw new Error("inconsistent state: valset front keys mismatch");
    }
    for (const val_id of pareto_ids) {
      if (!mapping_ids.has(val_id)) {
        throw new Error("inconsistent state: valset front keys mismatch");
      }
    }

    const objective_keys = new Set(Object.keys(this.objective_pareto_front));
    const objective_map_keys = new Set(this.program_at_pareto_front_objectives.keys());
    if (objective_keys.size !== objective_map_keys.size) {
      throw new Error("inconsistent state: objective front keys mismatch");
    }
    for (const objective_key of objective_keys) {
      if (!objective_map_keys.has(objective_key)) {
        throw new Error("inconsistent state: objective front keys mismatch");
      }
    }

    for (const front of this.program_at_pareto_front_valset.values()) {
      for (const prog_idx of front) {
        if (prog_idx >= n) {
          throw new Error("Program index in valset pareto front exceeds number of program candidates");
        }
      }
    }

    return true;
  }

  add_budget_hook(hook: (new_total: number, delta: number) => void): void {
    if (!this._budget_hooks) {
      this._budget_hooks = [];
    }
    this._budget_hooks.push(hook);
  }

  increment_evals(count: number): void {
    this.total_num_evals += count;
    if (this._budget_hooks) {
      for (const hook of this._budget_hooks) {
        hook(this.total_num_evals, count);
      }
    }
  }

  static _aggregate_objective_scores<DataId extends string | number>(
    val_objective_scores: Map<DataId, ObjectiveScores> | null,
  ): ObjectiveScores {
    if (!val_objective_scores) {
      return {};
    }
    const totals = new Map<string, number>();
    const counts = new Map<string, number>();
    for (const objective_dict of val_objective_scores.values()) {
      for (const [objective, score] of Object.entries(objective_dict)) {
        totals.set(objective, (totals.get(objective) ?? 0) + score);
        counts.set(objective, (counts.get(objective) ?? 0) + 1);
      }
    }
    const out: ObjectiveScores = {};
    for (const [objective, total] of totals.entries()) {
      const count = counts.get(objective);
      if (count && count > 0) {
        out[objective] = total / count;
      }
    }
    return out;
  }

  get_program_average_val_subset(program_idx: number): [number, number] {
    const scores = this.prog_candidate_val_subscores[program_idx];
    if (!scores || scores.size === 0) {
      return [Number.NEGATIVE_INFINITY, 0];
    }
    let total = 0;
    for (const score of scores.values()) {
      total += score;
    }
    return [total / scores.size, scores.size];
  }

  get valset_evaluations(): Map<DataId, number[]> {
    const result = new Map<DataId, number[]>();
    for (let program_idx = 0; program_idx < this.prog_candidate_val_subscores.length; program_idx += 1) {
      const val_scores = this.prog_candidate_val_subscores[program_idx];
      if (!val_scores) {
        continue;
      }
      for (const val_id of val_scores.keys()) {
        const existing = result.get(val_id);
        if (existing) {
          existing.push(program_idx);
        } else {
          result.set(val_id, [program_idx]);
        }
      }
    }
    return result;
  }

  get program_full_scores_val_set(): number[] {
    return this.prog_candidate_val_subscores.map((_, idx) => this.get_program_average_val_subset(idx)[0]);
  }

  get per_program_tracked_scores(): number[] {
    return this.prog_candidate_val_subscores.map((_, idx) => this.get_program_average_val_subset(idx)[0]);
  }

  private _update_objective_pareto_front(objective_scores: ObjectiveScores, program_idx: number): void {
    for (const [objective, score] of Object.entries(objective_scores)) {
      const prev_score = this.objective_pareto_front[objective] ?? Number.NEGATIVE_INFINITY;
      if (score > prev_score) {
        this.objective_pareto_front[objective] = score;
        this.program_at_pareto_front_objectives.set(objective, new Set([program_idx]));
      } else if (score === prev_score) {
        const front = this.program_at_pareto_front_objectives.get(objective) ?? new Set<number>();
        front.add(program_idx);
        this.program_at_pareto_front_objectives.set(objective, front);
      }
    }
  }

  private _update_pareto_front_for_val_id(
    val_id: DataId,
    score: number,
    program_idx: number,
    output: RolloutOutput | null,
  ): void {
    const prev_score = this.pareto_front_valset.get(val_id) ?? Number.NEGATIVE_INFINITY;
    if (score > prev_score) {
      this.pareto_front_valset.set(val_id, score);
      this.program_at_pareto_front_valset.set(val_id, new Set([program_idx]));
      if (this.best_outputs_valset && output !== null) {
        this.best_outputs_valset.set(val_id, [[program_idx, output]]);
      }
      return;
    }
    if (score === prev_score) {
      const front = this.program_at_pareto_front_valset.get(val_id) ?? new Set<number>();
      front.add(program_idx);
      this.program_at_pareto_front_valset.set(val_id, front);
      if (this.best_outputs_valset && output !== null) {
        const entries = this.best_outputs_valset.get(val_id) ?? [];
        entries.push([program_idx, output]);
        this.best_outputs_valset.set(val_id, entries);
      }
    }
  }

  update_state_with_new_program(opts: {
    parent_program_idx: number[];
    new_program: Candidate;
    valset_evaluation: ValsetEvaluation<RolloutOutput, DataId>;
    run_dir: string | null;
    num_metric_calls_by_discovery_of_new_program: number;
  }): number {
    if (this.frontier_type !== "instance") {
      throw new Error(`frontier_type "${this.frontier_type}" not supported in v1`);
    }

    const new_program_idx = this.program_candidates.length;
    this.program_candidates.push({ ...opts.new_program });
    this.num_metric_calls_by_discovery.push(opts.num_metric_calls_by_discovery_of_new_program);

    let max_predictor_id = 0;
    for (const parent_idx of opts.parent_program_idx) {
      const parent_next = this.named_predictor_id_to_update_next_for_program_candidate[parent_idx] ?? 0;
      if (parent_next > max_predictor_id) {
        max_predictor_id = parent_next;
      }
    }
    this.named_predictor_id_to_update_next_for_program_candidate.push(max_predictor_id);
    this.parent_program_for_candidate.push([...opts.parent_program_idx]);

    const valset_scores = new Map(opts.valset_evaluation.scores_by_val_id);
    this.prog_candidate_val_subscores.push(valset_scores);
    const objective_scores = GEPAState._aggregate_objective_scores(opts.valset_evaluation.objective_scores_by_val_id);
    this.prog_candidate_objective_scores.push(objective_scores);

    for (const [val_id, score] of valset_scores.entries()) {
      const output = opts.valset_evaluation.outputs_by_val_id.get(val_id) ?? null;
      this._update_pareto_front_for_val_id(val_id, score, new_program_idx, output);
    }

    this._update_objective_pareto_front(objective_scores, new_program_idx);

    return new_program_idx;
  }

  private _get_pareto_front_mapping(frontier_type: SupportedFrontierType): Map<DataId, Set<number>> {
    if (frontier_type === "instance") {
      return new Map(Array.from(this.program_at_pareto_front_valset.entries()).map(([k, v]) => [k, new Set(v)]));
    }
    throw new Error(`frontier_type "${frontier_type}" not supported in v1`);
  }

  get_pareto_front_mapping(): Map<DataId, Set<number>> {
    return this._get_pareto_front_mapping(this.frontier_type);
  }

  cached_evaluate(
    candidate: Candidate,
    example_ids: DataId[],
    fetcher: (example_ids: DataId[]) => unknown,
    evaluator: (
      batch: unknown,
      candidate: Candidate,
    ) => [RolloutOutput[], number[], Array<Record<string, number>> | null],
  ): [number[], number] {
    const [, scores_by_id, , num_actual_evals] = this.cached_evaluate_full(candidate, example_ids, fetcher, evaluator);
    const ordered = example_ids.map((eid) => {
      const score = scores_by_id.get(eid);
      if (score === undefined) {
        throw new Error(`Missing score for example id: ${String(eid)}`);
      }
      return score;
    });
    return [ordered, num_actual_evals];
  }

  cached_evaluate_full(
    candidate: Candidate,
    example_ids: DataId[],
    fetcher: (example_ids: DataId[]) => unknown,
    evaluator: (
      batch: unknown,
      candidate: Candidate,
    ) => [RolloutOutput[], number[], Array<Record<string, number>> | null],
  ): [Map<DataId, RolloutOutput>, Map<DataId, number>, Map<DataId, Record<string, number>> | null, number] {
    const batch = fetcher(example_ids);
    const [outputs, scores, objective_scores] = evaluator(batch, candidate);
    const outputs_by_id = new Map<DataId, RolloutOutput>();
    const scores_by_id = new Map<DataId, number>();
    const objective_by_id = objective_scores ? new Map<DataId, Record<string, number>>() : null;

    for (let idx = 0; idx < example_ids.length; idx += 1) {
      const example_id = example_ids[idx];
      const output = outputs[idx];
      const score = scores[idx];
      if (example_id === undefined || output === undefined || score === undefined) {
        throw new Error("evaluator output shape does not match example_ids length");
      }
      outputs_by_id.set(example_id, output);
      scores_by_id.set(example_id, score);
      if (objective_by_id && objective_scores) {
        const objective = objective_scores[idx];
        if (objective === undefined) {
          throw new Error("objective_scores shape does not match example_ids length");
        }
        objective_by_id.set(example_id, objective);
      }
    }

    return [outputs_by_id, scores_by_id, objective_by_id, example_ids.length];
  }
}

export function initialize_gepa_state<RolloutOutput = unknown, DataId extends string | number = string | number>(opts: {
  run_dir: string | null;
  logger: { log(message: string): void };
  seed_candidate: Candidate;
  seed_valset_evaluation: ValsetEvaluation<RolloutOutput, DataId>;
  track_best_outputs?: boolean;
  frontier_type?: SupportedFrontierType;
}): GEPAState<RolloutOutput, DataId> {
  const constructor_opts: {
    seed_candidate: Candidate;
    base_evaluation: ValsetEvaluation<RolloutOutput, DataId>;
    track_best_outputs?: boolean;
    frontier_type?: SupportedFrontierType;
  } = {
    seed_candidate: opts.seed_candidate,
    base_evaluation: opts.seed_valset_evaluation,
  };
  if (opts.track_best_outputs !== undefined) {
    constructor_opts.track_best_outputs = opts.track_best_outputs;
  }
  if (opts.frontier_type !== undefined) {
    constructor_opts.frontier_type = opts.frontier_type;
  }
  const gepa_state = new GEPAState<RolloutOutput, DataId>(constructor_opts);

  gepa_state.num_full_ds_evals = 1;
  gepa_state.total_num_evals = opts.seed_valset_evaluation.scores_by_val_id.size;
  return gepa_state;
}
