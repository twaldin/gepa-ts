import { SINGLE_INSTANCE_SENTINEL, type Candidate } from "./types";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";

export type ObjectiveScores = Record<string, number>;
export type SupportedFrontierType = "instance" | "objective" | "hybrid" | "cartesian";

export type CachedEvaluation<RolloutOutput = unknown> = {
  output: RolloutOutput;
  score: number;
  objective_scores: ObjectiveScores | null;
};

type SerializedEvaluationCache<RolloutOutput, DataId extends string | number> = Array<{
  candidate_hash: string;
  example_id: DataId;
  value: CachedEvaluation<RolloutOutput>;
}>;

function candidate_hash(candidate: Candidate): string {
  const sorted_items = Object.entries(candidate).sort(([a], [b]) => a.localeCompare(b));
  const python_json = `[${sorted_items.map(([key, value]) => `[${JSON.stringify(key)}, ${JSON.stringify(value)}]`).join(", ")}]`;
  return createHash("sha256").update(python_json).digest("hex");
}

function cache_key(candidate_hash_value: string, example_id: string | number): string {
  return JSON.stringify([candidate_hash_value, example_id]);
}

export class EvaluationCache<RolloutOutput = unknown, DataId extends string | number = string | number> {
  private readonly cache = new Map<string, { candidate_hash: string; example_id: DataId; value: CachedEvaluation<RolloutOutput> }>();

  get(candidate: Candidate, example_id: DataId): CachedEvaluation<RolloutOutput> | null {
    return this.cache.get(cache_key(candidate_hash(candidate), example_id))?.value ?? null;
  }

  put(
    candidate: Candidate,
    example_id: DataId,
    output: RolloutOutput,
    score: number,
    objective_scores: ObjectiveScores | null = null,
  ): void {
    const hash = candidate_hash(candidate);
    this.cache.set(cache_key(hash, example_id), {
      candidate_hash: hash,
      example_id,
      value: { output, score, objective_scores },
    });
  }

  get_batch(
    candidate: Candidate,
    example_ids: DataId[],
  ): [Map<DataId, CachedEvaluation<RolloutOutput>>, DataId[]] {
    const hash = candidate_hash(candidate);
    const cached = new Map<DataId, CachedEvaluation<RolloutOutput>>();
    const uncached: DataId[] = [];
    for (const example_id of example_ids) {
      const entry = this.cache.get(cache_key(hash, example_id));
      if (entry) {
        cached.set(example_id, entry.value);
      } else {
        uncached.push(example_id);
      }
    }
    return [cached, uncached];
  }

  put_batch(
    candidate: Candidate,
    example_ids: DataId[],
    outputs: RolloutOutput[],
    scores: number[],
    objective_scores_list: ObjectiveScores[] | null = null,
  ): void {
    for (let idx = 0; idx < example_ids.length; idx += 1) {
      const example_id = example_ids[idx];
      const output = outputs[idx];
      const score = scores[idx];
      if (example_id === undefined || output === undefined || score === undefined) {
        throw new Error("cache put_batch shape does not match example_ids length");
      }
      this.put(candidate, example_id, output, score, objective_scores_list?.[idx] ?? null);
    }
  }

  async evaluate_with_cache_full(
    candidate: Candidate,
    example_ids: DataId[],
    fetcher: (example_ids: DataId[]) => unknown | Promise<unknown>,
    evaluator: (
      batch: unknown,
      candidate: Candidate,
      example_ids: DataId[],
    ) => [RolloutOutput[], number[], ObjectiveScores[] | null] | Promise<[RolloutOutput[], number[], ObjectiveScores[] | null]>,
  ): Promise<[Map<DataId, RolloutOutput>, Map<DataId, number>, Map<DataId, ObjectiveScores> | null, number]> {
    const [cached, uncached_ids] = this.get_batch(candidate, example_ids);
    const outputs_by_id = new Map<DataId, RolloutOutput>();
    const scores_by_id = new Map<DataId, number>();
    let objective_by_id: Map<DataId, ObjectiveScores> | null = null;

    for (const [example_id, entry] of cached.entries()) {
      outputs_by_id.set(example_id, entry.output);
      scores_by_id.set(example_id, entry.score);
      if (entry.objective_scores !== null) {
        if (objective_by_id === null) {
          objective_by_id = new Map();
        }
        objective_by_id.set(example_id, entry.objective_scores);
      }
    }

    if (uncached_ids.length > 0) {
      const batch = await fetcher(uncached_ids);
      const [outputs, scores, objective_scores] = await evaluator(batch, candidate, uncached_ids);
      for (let idx = 0; idx < uncached_ids.length; idx += 1) {
        const example_id = uncached_ids[idx];
        const output = outputs[idx];
        const score = scores[idx];
        if (example_id === undefined || output === undefined || score === undefined) {
          throw new Error("evaluator output shape does not match uncached example_ids length");
        }
        outputs_by_id.set(example_id, output);
        scores_by_id.set(example_id, score);
        const objective_scores_for_example = objective_scores?.[idx] ?? null;
        if (objective_scores_for_example !== null) {
          if (objective_by_id === null) {
            objective_by_id = new Map();
          }
          objective_by_id.set(example_id, objective_scores_for_example);
        }
      }
      this.put_batch(candidate, uncached_ids, outputs, scores, objective_scores);
    }

    return [outputs_by_id, scores_by_id, objective_by_id, uncached_ids.length];
  }

  serialize(): SerializedEvaluationCache<RolloutOutput, DataId> {
    return Array.from(this.cache.values()).map((entry) => ({
      candidate_hash: entry.candidate_hash,
      example_id: entry.example_id,
      value: {
        output: entry.value.output,
        score: entry.value.score,
        objective_scores: entry.value.objective_scores === null ? null : { ...entry.value.objective_scores },
      },
    }));
  }

  static from_serialized<RolloutOutput = unknown, DataId extends string | number = string | number>(
    entries: SerializedEvaluationCache<RolloutOutput, DataId>,
  ): EvaluationCache<RolloutOutput, DataId> {
    const cache = new EvaluationCache<RolloutOutput, DataId>();
    for (const entry of entries) {
      cache.cache.set(cache_key(entry.candidate_hash, entry.example_id), {
        candidate_hash: entry.candidate_hash,
        example_id: entry.example_id,
        value: {
          output: entry.value.output,
          score: entry.value.score,
          objective_scores: entry.value.objective_scores === null ? null : { ...entry.value.objective_scores },
        },
      });
    }
    return cache;
  }
}

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

export const SINGLE_INSTANCE_BEST_EVALS_KEY = "__single_instance__";
const GEPA_STATE_JSON = "gepa_state.json";

type Entry<K, V> = [K, V];
type SerializedGEPAState<RolloutOutput, DataId extends string | number> = {
  validation_schema_version: 1;
  program_candidates: Candidate[];
  prog_candidate_val_subscores: Array<Array<Entry<DataId, number>>>;
  prog_candidate_objective_scores: ObjectiveScores[];
  parent_program_for_candidate: (number | null)[][];
  pareto_front_valset: Array<Entry<DataId, number>>;
  program_at_pareto_front_valset: Array<Entry<DataId, number[]>>;
  objective_pareto_front: ObjectiveScores;
  program_at_pareto_front_objectives: Array<Entry<string, number[]>>;
  pareto_front_cartesian: Array<Entry<[DataId, string], number>>;
  program_at_pareto_front_cartesian: Array<Entry<[DataId, string], number[]>>;
  list_of_named_predictors: string[];
  named_predictor_id_to_update_next_for_program_candidate: number[];
  i: number;
  num_full_ds_evals: number;
  total_num_evals: number;
  num_metric_calls_by_discovery: number[];
  full_program_trace: Record<string, unknown>[];
  best_outputs_valset: Array<Entry<DataId, Array<[number, RolloutOutput]>>> | null;
  frontier_type: SupportedFrontierType;
  adapter_state: Record<string, unknown>;
  best_example_evals: Array<Entry<DataId, Array<{ score: number; side_info: Record<string, unknown> }>>>;
  evaluation_cache?: SerializedEvaluationCache<RolloutOutput, DataId> | null;
};

function map_entries<K, V>(map: Map<K, V>): Array<Entry<K, V>> {
  return Array.from(map.entries());
}

function set_map_entries<K>(map: Map<K, Set<number>>): Array<Entry<K, number[]>> {
  return Array.from(map.entries()).map(([key, values]) => [key, Array.from(values)]);
}

function write_json_file(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`);
}

function write_eval_outputs_to_directory<RolloutOutput, DataId extends string | number>(
  outputs: Map<DataId, RolloutOutput>,
  output_dir: string,
): void {
  for (const [val_id, output] of outputs.entries()) {
    const task_dir = `${output_dir}/task_${String(val_id)}`;
    mkdirSync(task_dir, { recursive: true });
    write_json_file(`${task_dir}/iter_0_prog_0.json`, output);
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
  pareto_front_cartesian: Map<[DataId, string], number>;
  program_at_pareto_front_cartesian: Map<[DataId, string], Set<number>>;
  list_of_named_predictors: string[];
  named_predictor_id_to_update_next_for_program_candidate: number[];
  i: number;
  num_full_ds_evals: number;
  total_num_evals: number;
  num_metric_calls_by_discovery: number[];
  full_program_trace: Record<string, unknown>[];
  best_outputs_valset: Map<DataId, [number, RolloutOutput][]> | null;
  frontier_type: SupportedFrontierType;
  evaluation_cache: EvaluationCache<RolloutOutput, DataId> | null;
  adapter_state: Record<string, unknown>;
  best_example_evals: Map<DataId, Array<{ score: number; side_info: Record<string, unknown> }>>;

  private _budget_hooks?: Array<(new_total: number, delta: number) => void>;

  constructor(opts: {
    seed_candidate: Candidate;
    base_evaluation: ValsetEvaluation<RolloutOutput, DataId>;
    track_best_outputs?: boolean;
    frontier_type?: SupportedFrontierType;
    evaluation_cache?: EvaluationCache<RolloutOutput, DataId> | null;
  }) {
    this.frontier_type = opts.frontier_type ?? "instance";

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

    if (
      (this.frontier_type === "objective" || this.frontier_type === "hybrid" || this.frontier_type === "cartesian") &&
      opts.base_evaluation.objective_scores_by_val_id === null
    ) {
      throw new Error(
        `frontier_type='${this.frontier_type}' requires objective_scores to be provided by the evaluator, but none were found. Use an evaluator that returns objective_scores or use frontier_type='instance'.`,
      );
    }

    this.pareto_front_cartesian = new Map();
    this.program_at_pareto_front_cartesian = new Map();
    if (this.frontier_type === "cartesian" && opts.base_evaluation.objective_scores_by_val_id !== null) {
      for (const [val_id, objective_scores] of opts.base_evaluation.objective_scores_by_val_id.entries()) {
        for (const [objective, objective_score] of Object.entries(objective_scores)) {
          const key: [DataId, string] = [val_id, objective];
          this.pareto_front_cartesian.set(key, objective_score);
          this.program_at_pareto_front_cartesian.set(key, new Set([0]));
        }
      }
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
    this.evaluation_cache = opts.evaluation_cache ?? null;
    this.adapter_state = {};
    this.best_example_evals = new Map();
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

  record_example_eval(val_id: DataId, score: number, side_info: Record<string, unknown>, k: number): void {
    const key = (String(val_id) === String(SINGLE_INSTANCE_SENTINEL)
      ? SINGLE_INSTANCE_BEST_EVALS_KEY
      : val_id) as DataId;
    const existing = this.best_example_evals.get(key) ?? [];
    existing.push({ score, side_info });
    existing.sort((a, b) => b.score - a.score);
    this.best_example_evals.set(key, existing.slice(0, k));
  }

  save(run_dir: string | null): void {
    if (run_dir === null) {
      return;
    }
    mkdirSync(run_dir, { recursive: true });
    const serialized: SerializedGEPAState<RolloutOutput, DataId> = {
      validation_schema_version: 1,
      program_candidates: this.program_candidates.map((candidate) => ({ ...candidate })),
      prog_candidate_val_subscores: this.prog_candidate_val_subscores.map((scores) => map_entries(scores)),
      prog_candidate_objective_scores: this.prog_candidate_objective_scores.map((scores) => ({ ...scores })),
      parent_program_for_candidate: this.parent_program_for_candidate.map((parents) => [...parents]),
      pareto_front_valset: map_entries(this.pareto_front_valset),
      program_at_pareto_front_valset: set_map_entries(this.program_at_pareto_front_valset),
      objective_pareto_front: { ...this.objective_pareto_front },
      program_at_pareto_front_objectives: set_map_entries(this.program_at_pareto_front_objectives),
      pareto_front_cartesian: map_entries(this.pareto_front_cartesian),
      program_at_pareto_front_cartesian: set_map_entries(this.program_at_pareto_front_cartesian),
      list_of_named_predictors: [...this.list_of_named_predictors],
      named_predictor_id_to_update_next_for_program_candidate: [
        ...this.named_predictor_id_to_update_next_for_program_candidate,
      ],
      i: this.i,
      num_full_ds_evals: this.num_full_ds_evals,
      total_num_evals: this.total_num_evals,
      num_metric_calls_by_discovery: [...this.num_metric_calls_by_discovery],
      full_program_trace: this.full_program_trace.map((trace) => ({ ...trace })),
      best_outputs_valset: this.best_outputs_valset === null
        ? null
        : map_entries(this.best_outputs_valset).map(([key, outputs]) => [key, outputs.map(([idx, output]) => [idx, output])]),
      frontier_type: this.frontier_type,
      adapter_state: { ...this.adapter_state },
      best_example_evals: map_entries(this.best_example_evals).map(([key, evals]) => [
        key,
        evals.map((entry) => ({ score: entry.score, side_info: { ...entry.side_info } })),
      ]),
      evaluation_cache: this.evaluation_cache?.serialize() ?? null,
    };

    write_json_file(`${run_dir}/${GEPA_STATE_JSON}`, serialized);
    if (this.full_program_trace.length > 0) {
      write_json_file(`${run_dir}/run_log.json`, this.full_program_trace);
    }
    if (this.program_candidates.length > 0) {
      write_json_file(`${run_dir}/candidates.json`, this.program_candidates);
    }
  }

  static load<RolloutOutput = unknown, DataId extends string | number = string | number>(
    run_dir: string,
  ): GEPAState<RolloutOutput, DataId> {
    const data = JSON.parse(readFileSync(`${run_dir}/${GEPA_STATE_JSON}`, "utf8")) as SerializedGEPAState<
      RolloutOutput,
      DataId
    >;
    const state = Object.create(GEPAState.prototype) as GEPAState<RolloutOutput, DataId>;

    state.program_candidates = data.program_candidates.map((candidate) => ({ ...candidate }));
    state.prog_candidate_val_subscores = data.prog_candidate_val_subscores.map((entries) => new Map(entries));
    state.prog_candidate_objective_scores = data.prog_candidate_objective_scores.map((scores) => ({ ...scores }));
    state.parent_program_for_candidate = data.parent_program_for_candidate.map((parents) => [...parents]);
    state.pareto_front_valset = new Map(data.pareto_front_valset);
    state.program_at_pareto_front_valset = new Map(
      data.program_at_pareto_front_valset.map(([key, values]) => [key, new Set(values)]),
    );
    state.objective_pareto_front = { ...data.objective_pareto_front };
    state.program_at_pareto_front_objectives = new Map(
      data.program_at_pareto_front_objectives.map(([key, values]) => [key, new Set(values)]),
    );
    state.pareto_front_cartesian = new Map(data.pareto_front_cartesian ?? []);
    state.program_at_pareto_front_cartesian = new Map(
      (data.program_at_pareto_front_cartesian ?? []).map(([key, values]) => [key, new Set(values)]),
    );
    state.list_of_named_predictors = [...data.list_of_named_predictors];
    state.named_predictor_id_to_update_next_for_program_candidate = [
      ...data.named_predictor_id_to_update_next_for_program_candidate,
    ];
    state.i = data.i;
    state.num_full_ds_evals = data.num_full_ds_evals;
    state.total_num_evals = data.total_num_evals;
    state.num_metric_calls_by_discovery = [...data.num_metric_calls_by_discovery];
    state.full_program_trace = data.full_program_trace.map((trace) => ({ ...trace }));
    state.best_outputs_valset = data.best_outputs_valset === null
      ? null
      : new Map(data.best_outputs_valset.map(([key, outputs]) => [key, outputs.map(([idx, output]) => [idx, output])]));
    state.frontier_type = data.frontier_type;
    state.evaluation_cache = data.evaluation_cache
      ? EvaluationCache.from_serialized<RolloutOutput, DataId>(data.evaluation_cache)
      : null;
    state.adapter_state = { ...data.adapter_state };
    state.best_example_evals = new Map(
      data.best_example_evals.map(([key, evals]) => [
        key,
        evals.map((entry) => ({ score: entry.score, side_info: { ...entry.side_info } })),
      ]),
    );
    state.is_consistent();
    return state;
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
    run_dir: string | null,
  ): void {
    const prev_score = this.pareto_front_valset.get(val_id) ?? Number.NEGATIVE_INFINITY;
    if (score > prev_score) {
      this.pareto_front_valset.set(val_id, score);
      this.program_at_pareto_front_valset.set(val_id, new Set([program_idx]));
      if (this.best_outputs_valset && output !== null) {
        this.best_outputs_valset.set(val_id, [[program_idx, output]]);
        if (run_dir !== null) {
          const task_dir = `${run_dir}/generated_best_outputs_valset/task_${String(val_id)}`;
          mkdirSync(task_dir, { recursive: true });
          write_json_file(`${task_dir}/iter_${this.i + 1}_prog_${program_idx}.json`, output);
        }
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

  private _update_pareto_front_for_cartesian(
    val_id: DataId,
    objective: string,
    objective_score: number,
    program_idx: number,
  ): void {
    let matched_key: [DataId, string] | null = null;
    for (const key of this.pareto_front_cartesian.keys()) {
      if (String(key[0]) === String(val_id) && key[1] === objective) {
        matched_key = key;
        break;
      }
    }
    const key = matched_key ?? [val_id, objective] as [DataId, string];
    const prev_score = this.pareto_front_cartesian.get(key) ?? Number.NEGATIVE_INFINITY;
    if (objective_score > prev_score) {
      this.pareto_front_cartesian.set(key, objective_score);
      this.program_at_pareto_front_cartesian.set(key, new Set([program_idx]));
      return;
    }
    if (objective_score === prev_score) {
      const front = this.program_at_pareto_front_cartesian.get(key) ?? new Set<number>();
      front.add(program_idx);
      this.program_at_pareto_front_cartesian.set(key, front);
    }
  }

  update_state_with_new_program(opts: {
    parent_program_idx: number[];
    new_program: Candidate;
    valset_evaluation: ValsetEvaluation<RolloutOutput, DataId>;
    run_dir: string | null;
    num_metric_calls_by_discovery_of_new_program: number;
  }): number {
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
      this._update_pareto_front_for_val_id(val_id, score, new_program_idx, output, opts.run_dir);
    }

    this._update_objective_pareto_front(objective_scores, new_program_idx);
    if (
      (this.frontier_type === "objective" || this.frontier_type === "hybrid" || this.frontier_type === "cartesian") &&
      opts.valset_evaluation.objective_scores_by_val_id === null
    ) {
      throw new Error(
        `frontier_type='${this.frontier_type}' requires objective_scores to be provided by the evaluator, but none were found in the evaluation result.`,
      );
    }
    if (this.frontier_type === "cartesian" && opts.valset_evaluation.objective_scores_by_val_id !== null) {
      for (const [val_id, objective_scores] of opts.valset_evaluation.objective_scores_by_val_id.entries()) {
        for (const [objective, objective_score] of Object.entries(objective_scores)) {
          this._update_pareto_front_for_cartesian(val_id, objective, objective_score, new_program_idx);
        }
      }
    }
    this.save(opts.run_dir);

    return new_program_idx;
  }

  private _get_pareto_front_mapping(frontier_type: SupportedFrontierType): Map<unknown, Set<number>> {
    if (frontier_type === "instance") {
      return new Map(Array.from(this.program_at_pareto_front_valset.entries()).map(([k, v]) => [k, new Set(v)]));
    }
    if (frontier_type === "objective") {
      return new Map(Array.from(this.program_at_pareto_front_objectives.entries()).map(([k, v]) => [k as DataId, new Set(v)]));
    }
    if (frontier_type === "hybrid") {
      const combined = new Map<unknown, Set<number>>();
      for (const [val_id, front] of this.program_at_pareto_front_valset.entries()) {
        combined.set(["val_id", val_id], new Set(front));
      }
      for (const [objective, front] of this.program_at_pareto_front_objectives.entries()) {
        combined.set(["objective", objective], new Set(front));
      }
      return combined;
    }
    if (frontier_type === "cartesian") {
      const cartesian = new Map<unknown, Set<number>>();
      for (const [key, front] of this.program_at_pareto_front_cartesian.entries()) {
        cartesian.set(["cartesian", key[0], key[1]], new Set(front));
      }
      return cartesian;
    }
    throw new Error(`Unknown frontier_type: ${frontier_type}`);
  }

  get_pareto_front_mapping(): Map<unknown, Set<number>> {
    return this._get_pareto_front_mapping(this.frontier_type);
  }

  async cached_evaluate(
    candidate: Candidate,
    example_ids: DataId[],
    fetcher: (example_ids: DataId[]) => unknown,
    evaluator: (
      batch: unknown,
      candidate: Candidate,
      example_ids: DataId[],
    ) => [RolloutOutput[], number[], Array<Record<string, number>> | null] | Promise<[RolloutOutput[], number[], Array<Record<string, number>> | null]>,
  ): Promise<[number[], number]> {
    const [, scores_by_id, , num_actual_evals] = await this.cached_evaluate_full(candidate, example_ids, fetcher, evaluator);
    const ordered = example_ids.map((eid) => {
      const score = scores_by_id.get(eid);
      if (score === undefined) {
        throw new Error(`Missing score for example id: ${String(eid)}`);
      }
      return score;
    });
    return [ordered, num_actual_evals];
  }

  async cached_evaluate_full(
    candidate: Candidate,
    example_ids: DataId[],
    fetcher: (example_ids: DataId[]) => unknown | Promise<unknown>,
    evaluator: (
      batch: unknown,
      candidate: Candidate,
      example_ids: DataId[],
    ) => [RolloutOutput[], number[], Array<Record<string, number>> | null] | Promise<[RolloutOutput[], number[], Array<Record<string, number>> | null]>,
  ): Promise<[Map<DataId, RolloutOutput>, Map<DataId, number>, Map<DataId, Record<string, number>> | null, number]> {
    if (this.evaluation_cache !== null) {
      return this.evaluation_cache.evaluate_with_cache_full(candidate, example_ids, fetcher, evaluator);
    }
    const batch = await fetcher(example_ids);
    const [outputs, scores, objective_scores] = await evaluator(batch, candidate, example_ids);
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
  evaluation_cache?: EvaluationCache<RolloutOutput, DataId> | null;
}): GEPAState<RolloutOutput, DataId> {
  if (opts.run_dir !== null && existsSync(`${opts.run_dir}/${GEPA_STATE_JSON}`)) {
    opts.logger.log("Loading gepa state from run dir");
    const gepa_state = GEPAState.load<RolloutOutput, DataId>(opts.run_dir);
    if (gepa_state.frontier_type !== (opts.frontier_type ?? "instance")) {
      throw new Error(
        `Frontier type mismatch: requested '${opts.frontier_type ?? "instance"}' but loaded state has '${gepa_state.frontier_type}'. Use a different run_dir or match the frontier_type parameter.`,
      );
    }
    if (opts.evaluation_cache === undefined || opts.evaluation_cache === null) {
      gepa_state.evaluation_cache = null;
    } else if (gepa_state.evaluation_cache === null) {
      gepa_state.evaluation_cache = opts.evaluation_cache;
    }
    return gepa_state;
  }

  if (opts.run_dir !== null) {
    write_eval_outputs_to_directory(
      opts.seed_valset_evaluation.outputs_by_val_id,
      `${opts.run_dir}/generated_best_outputs_valset`,
    );
  }

  const constructor_opts: {
    seed_candidate: Candidate;
    base_evaluation: ValsetEvaluation<RolloutOutput, DataId>;
    track_best_outputs?: boolean;
    frontier_type?: SupportedFrontierType;
    evaluation_cache?: EvaluationCache<RolloutOutput, DataId> | null;
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
  if (opts.evaluation_cache !== undefined) {
    constructor_opts.evaluation_cache = opts.evaluation_cache;
  }
  const gepa_state = new GEPAState<RolloutOutput, DataId>(constructor_opts);

  gepa_state.num_full_ds_evals = 1;
  gepa_state.total_num_evals = opts.seed_valset_evaluation.scores_by_val_id.size;
  return gepa_state;
}
