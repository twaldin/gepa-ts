import type { DataId, ProgramIdx } from './types.js';
import type { GEPAState } from './state.js';
import { candidate_tree_dot_from_data, candidate_tree_html_from_data } from './visualization.js';

export interface GEPAResult {
  readonly candidates: Array<Record<string, string>>;
  readonly parents: Array<Array<ProgramIdx | null>>;
  readonly val_aggregate_scores: number[];
  readonly val_subscores: Array<Map<DataId, number>>;
  readonly per_val_instance_best_candidates: Map<DataId, Set<ProgramIdx>>;
  readonly discovery_eval_counts: number[];
  readonly val_aggregate_subscores: Array<Record<string, number>> | null;
  readonly per_objective_best_candidates: Map<string, Set<ProgramIdx>> | null;
  readonly objective_pareto_front: Record<string, number> | null;
  readonly best_outputs_valset: Map<DataId, [ProgramIdx, unknown][]> | null;
  readonly total_metric_calls: number | null;
  readonly num_full_val_evals: number | null;
  readonly run_dir: string | null;
  readonly seed: number | null;
  readonly _str_candidate_key: string | null;

  readonly num_candidates: number;
  readonly num_val_instances: number;
  readonly best_idx: number;
  readonly best_candidate: string | Record<string, string>;
  readonly best_refiner_prompt: string | undefined;
  to_dict(): Record<string, unknown>;
  candidate_tree_dot(): string;
  candidate_tree_html(): string;
}

class GEPAResultImpl implements GEPAResult {
  readonly candidates: Array<Record<string, string>>;
  readonly parents: Array<Array<ProgramIdx | null>>;
  readonly val_aggregate_scores: number[];
  readonly val_subscores: Array<Map<DataId, number>>;
  readonly per_val_instance_best_candidates: Map<DataId, Set<ProgramIdx>>;
  readonly discovery_eval_counts: number[];
  readonly val_aggregate_subscores: Array<Record<string, number>> | null;
  readonly per_objective_best_candidates: Map<string, Set<ProgramIdx>> | null;
  readonly objective_pareto_front: Record<string, number> | null;
  readonly best_outputs_valset: Map<DataId, [ProgramIdx, unknown][]> | null;
  readonly total_metric_calls: number | null;
  readonly num_full_val_evals: number | null;
  readonly run_dir: string | null;
  readonly seed: number | null;
  readonly _str_candidate_key: string | null;

  constructor(opts: {
    candidates: Array<Record<string, string>>;
    parents: Array<Array<ProgramIdx | null>>;
    val_aggregate_scores: number[];
    val_subscores: Array<Map<DataId, number>>;
    per_val_instance_best_candidates: Map<DataId, Set<ProgramIdx>>;
    discovery_eval_counts: number[];
    val_aggregate_subscores: Array<Record<string, number>> | null;
    per_objective_best_candidates: Map<string, Set<ProgramIdx>> | null;
    objective_pareto_front: Record<string, number> | null;
    best_outputs_valset: Map<DataId, [ProgramIdx, unknown][]> | null;
    total_metric_calls: number | null;
    num_full_val_evals: number | null;
    run_dir: string | null;
    seed: number | null;
    _str_candidate_key: string | null;
  }) {
    this.candidates = opts.candidates;
    this.parents = opts.parents;
    this.val_aggregate_scores = opts.val_aggregate_scores;
    this.val_subscores = opts.val_subscores;
    this.per_val_instance_best_candidates = opts.per_val_instance_best_candidates;
    this.discovery_eval_counts = opts.discovery_eval_counts;
    this.val_aggregate_subscores = opts.val_aggregate_subscores;
    this.per_objective_best_candidates = opts.per_objective_best_candidates;
    this.objective_pareto_front = opts.objective_pareto_front;
    this.best_outputs_valset = opts.best_outputs_valset;
    this.total_metric_calls = opts.total_metric_calls;
    this.num_full_val_evals = opts.num_full_val_evals;
    this.run_dir = opts.run_dir;
    this.seed = opts.seed;
    this._str_candidate_key = opts._str_candidate_key;
  }

  get num_candidates(): number {
    return this.candidates.length;
  }

  get num_val_instances(): number {
    return this.per_val_instance_best_candidates.size;
  }

  get best_idx(): number {
    const scores = this.val_aggregate_scores;
    return scores.reduce((best_i, score, i) => {
      const best_score = scores[best_i];
      return best_score !== undefined && score > best_score ? i : best_i;
    }, 0);
  }

  get best_candidate(): string | Record<string, string> {
    const cand = this.candidates[this.best_idx];
    if (cand === undefined) return {};
    if (this._str_candidate_key !== null) {
      const val = cand[this._str_candidate_key];
      if (val !== undefined) return val;
    }
    return cand;
  }

  get best_refiner_prompt(): string | undefined {
    return this.candidates[this.best_idx]?.['refiner_prompt'];
  }

  to_dict(): Record<string, unknown> {
    return {
      candidates: this.candidates.map((candidate) => ({ ...candidate })),
      parents: this.parents.map((parent_row) => [...parent_row]),
      val_aggregate_scores: [...this.val_aggregate_scores],
      val_subscores: this.val_subscores.map((scores) => Object.fromEntries(scores.entries())),
      best_outputs_valset: this.best_outputs_valset === null
        ? null
        : Object.fromEntries(
            Array.from(this.best_outputs_valset.entries()).map(([val_id, outputs]) => [
              String(val_id),
              outputs.map(([program_idx, output]) => [program_idx, output]),
            ]),
          ),
      per_val_instance_best_candidates: Object.fromEntries(
        Array.from(this.per_val_instance_best_candidates.entries()).map(([val_id, front]) => [
          String(val_id),
          [...front],
        ]),
      ),
      val_aggregate_subscores: this.val_aggregate_subscores === null
        ? null
        : this.val_aggregate_subscores.map((scores) => ({ ...scores })),
      per_objective_best_candidates: this.per_objective_best_candidates === null
        ? null
        : Object.fromEntries(
            Array.from(this.per_objective_best_candidates.entries()).map(([objective, front]) => [
              objective,
              [...front],
            ]),
          ),
      objective_pareto_front: this.objective_pareto_front === null ? null : { ...this.objective_pareto_front },
      discovery_eval_counts: [...this.discovery_eval_counts],
      total_metric_calls: this.total_metric_calls,
      num_full_val_evals: this.num_full_val_evals,
      run_dir: this.run_dir,
      seed: this.seed,
      _str_candidate_key: this._str_candidate_key,
      best_idx: this.best_idx,
      validation_schema_version: 2,
    };
  }

  candidate_tree_dot(): string {
    return candidate_tree_dot_from_data(
      this.candidates,
      this.parents,
      this.val_aggregate_scores,
      this.per_val_instance_best_candidates,
    );
  }

  candidate_tree_html(): string {
    return candidate_tree_html_from_data(
      this.candidates,
      this.parents,
      this.val_aggregate_scores,
      this.per_val_instance_best_candidates,
    );
  }
}

export function result_from_state(
  state: GEPAState,
  opts: { str_candidate_key?: string | null; run_dir?: string | null; seed?: number | null },
): GEPAResult {
  const objective_scores_list = state.prog_candidate_objective_scores.map((s) => ({ ...s }));
  const has_objective_scores = objective_scores_list.some((obj) => Object.keys(obj).length > 0);

  const per_objective_best: Map<string, Set<number>> = new Map(
    Array.from(state.program_at_pareto_front_objectives.entries()).map(([k, v]) => [k, new Set(v)]),
  );
  const objective_front = { ...state.objective_pareto_front };

  const best_outputs_valset: Map<DataId, [number, unknown][]> | null = state.best_outputs_valset
    ? new Map(
        Array.from(state.best_outputs_valset.entries()).map(([k, v]) => [k, v.map(([pi, o]) => [pi, o] as [number, unknown])]),
      )
    : null;

  return new GEPAResultImpl({
    candidates: state.program_candidates.map((c) => ({ ...c })),
    parents: state.parent_program_for_candidate.map((p) => [...p]),
    val_aggregate_scores: [...state.program_full_scores_val_set],
    val_subscores: state.prog_candidate_val_subscores.map((m) => new Map(m)),
    per_val_instance_best_candidates: new Map(
      Array.from(state.program_at_pareto_front_valset.entries()).map(([k, v]) => [k, new Set(v)]),
    ),
    val_aggregate_subscores: has_objective_scores ? objective_scores_list : null,
    per_objective_best_candidates: per_objective_best.size > 0 ? per_objective_best : null,
    objective_pareto_front: Object.keys(objective_front).length > 0 ? objective_front : null,
    discovery_eval_counts: [...state.num_metric_calls_by_discovery],
    best_outputs_valset,
    total_metric_calls: state.total_num_evals,
    num_full_val_evals: state.num_full_ds_evals,
    run_dir: opts.run_dir ?? null,
    seed: opts.seed ?? null,
    _str_candidate_key: opts.str_candidate_key ?? null,
  });
}

function numeric_key_if_possible(key: string): DataId {
  return /^-?\d+$/.test(key) ? Number(key) : key;
}

function map_from_record(value: unknown): Map<DataId, number> {
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    return new Map(Object.entries(value).map(([key, score]) => [numeric_key_if_possible(key), Number(score)]));
  }
  return new Map();
}

function best_outputs_from_record(value: unknown): Map<DataId, [number, unknown][]> | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  return new Map(
    Object.entries(value).map(([key, outputs]) => [
      numeric_key_if_possible(key),
      Array.isArray(outputs)
        ? outputs.map((entry) => {
            if (Array.isArray(entry)) {
              return [Number(entry[0]), entry[1]] as [number, unknown];
            }
            return [0, entry] as [number, unknown];
          })
        : [],
    ]),
  );
}

function candidate_fronts_from_record(value: unknown): Map<DataId, Set<number>> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return new Map();
  }
  return new Map(
    Object.entries(value).map(([key, front]) => [
      numeric_key_if_possible(key),
      new Set(Array.isArray(front) ? front.map((idx) => Number(idx)) : []),
    ]),
  );
}

function objective_fronts_from_record(value: unknown): Map<string, Set<number>> | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  return new Map(
    Object.entries(value).map(([key, front]) => [
      key,
      new Set(Array.isArray(front) ? front.map((idx) => Number(idx)) : []),
    ]),
  );
}

function record_from_unknown(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function common_result_opts(d: Record<string, unknown>): ConstructorParameters<typeof GEPAResultImpl>[0] {
  return {
    candidates: Array.isArray(d['candidates'])
      ? d['candidates'].map((candidate) => {
          const raw = record_from_unknown(candidate);
          return Object.fromEntries(Object.entries(raw).map(([key, value]) => [key, String(value)]));
        })
      : [],
    parents: Array.isArray(d['parents'])
      ? d['parents'].map((parent_row) => Array.isArray(parent_row)
          ? parent_row.map((idx) => idx === null ? null : Number(idx))
          : [])
      : [],
    val_aggregate_scores: Array.isArray(d['val_aggregate_scores'])
      ? d['val_aggregate_scores'].map((score) => Number(score))
      : [],
    val_subscores: [],
    per_val_instance_best_candidates: new Map(),
    discovery_eval_counts: Array.isArray(d['discovery_eval_counts'])
      ? d['discovery_eval_counts'].map((count) => Number(count))
      : [],
    val_aggregate_subscores: null,
    per_objective_best_candidates: null,
    objective_pareto_front: null,
    best_outputs_valset: null,
    total_metric_calls: d['total_metric_calls'] === undefined || d['total_metric_calls'] === null ? null : Number(d['total_metric_calls']),
    num_full_val_evals: d['num_full_val_evals'] === undefined || d['num_full_val_evals'] === null ? null : Number(d['num_full_val_evals']),
    run_dir: d['run_dir'] === undefined || d['run_dir'] === null ? null : String(d['run_dir']),
    seed: d['seed'] === undefined || d['seed'] === null ? null : Number(d['seed']),
    _str_candidate_key: d['_str_candidate_key'] === undefined || d['_str_candidate_key'] === null ? null : String(d['_str_candidate_key']),
  };
}

export function result_from_dict(d: Record<string, unknown>): GEPAResult {
  const version = Number(d['validation_schema_version'] ?? 0);
  if (version > 2) {
    throw new Error(`Unsupported GEPAResult validation schema version ${version}; max supported is 2`);
  }

  const opts = common_result_opts(d);
  if (version <= 1) {
    opts.val_subscores = Array.isArray(d['val_subscores'])
      ? d['val_subscores'].map((scores) => Array.isArray(scores)
          ? new Map(scores.map((score, idx) => [idx, Number(score)]))
          : map_from_record(scores))
      : [];
    opts.per_val_instance_best_candidates = Array.isArray(d['per_val_instance_best_candidates'])
      ? new Map(d['per_val_instance_best_candidates'].map((front, idx) => [
          idx,
          new Set(Array.isArray(front) ? front.map((program_idx) => Number(program_idx)) : []),
        ]))
      : candidate_fronts_from_record(d['per_val_instance_best_candidates']);
    opts.best_outputs_valset = Array.isArray(d['best_outputs_valset'])
      ? new Map(d['best_outputs_valset'].map((outputs, idx) => [
          idx,
          Array.isArray(outputs)
            ? outputs.map((entry) => Array.isArray(entry) ? [Number(entry[0]), entry[1]] as [number, unknown] : [0, entry] as [number, unknown])
            : [],
        ]))
      : best_outputs_from_record(d['best_outputs_valset']);
    return new GEPAResultImpl(opts);
  }

  opts.val_subscores = Array.isArray(d['val_subscores'])
    ? d['val_subscores'].map((scores) => map_from_record(scores))
    : [];
  opts.per_val_instance_best_candidates = candidate_fronts_from_record(d['per_val_instance_best_candidates']);
  opts.best_outputs_valset = best_outputs_from_record(d['best_outputs_valset']);
  opts.val_aggregate_subscores = Array.isArray(d['val_aggregate_subscores'])
    ? d['val_aggregate_subscores'].map((scores) => {
        const raw = record_from_unknown(scores);
        return Object.fromEntries(Object.entries(raw).map(([key, value]) => [key, Number(value)]));
      })
    : null;
  opts.per_objective_best_candidates = objective_fronts_from_record(d['per_objective_best_candidates']);
  opts.objective_pareto_front = d['objective_pareto_front'] === null || d['objective_pareto_front'] === undefined
    ? null
    : Object.fromEntries(Object.entries(record_from_unknown(d['objective_pareto_front'])).map(([key, value]) => [key, Number(value)]));
  return new GEPAResultImpl(opts);
}
