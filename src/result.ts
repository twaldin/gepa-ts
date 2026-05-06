import type { DataId, ProgramIdx } from './types.js';
import type { GEPAState } from './state.js';

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
