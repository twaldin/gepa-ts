import type { DataId, EvaluationPolicy, ExperimentTrackerProtocol, ProgramIdx } from '../types.js';
import type { LoggerProtocol } from './logger.js';

type ProgramStateLike = {
  i: number;
  total_num_evals: number;
  program_full_scores_val_set: number[];
  pareto_front_valset: Map<DataId, number>;
  program_at_pareto_front_valset: Map<DataId, Set<ProgramIdx>>;
  objective_pareto_front: Map<string, number> | Record<string, number>;
  program_at_pareto_front_objectives: Map<string, Set<ProgramIdx>> | Record<string, Set<ProgramIdx>>;
  prog_candidate_val_subscores: Array<Map<DataId, number>>;
  prog_candidate_objective_scores: Array<Record<string, number>>;
  parent_program_for_candidate: Array<Array<ProgramIdx | null>>;
};

type ValsetEvaluationLike = {
  scores_by_val_id: Map<DataId, number>;
};

function objective_entries(objectives: Map<string, number> | Record<string, number>): Array<[string, number]> {
  return objectives instanceof Map ? [...objectives.entries()] : Object.entries(objectives);
}

function objective_front_set(
  fronts: Map<string, Set<ProgramIdx>> | Record<string, Set<ProgramIdx>>,
  objective: string,
): Set<ProgramIdx> {
  if (fronts instanceof Map) {
    return fronts.get(objective) ?? new Set<ProgramIdx>();
  }
  return fronts[objective] ?? new Set<ProgramIdx>();
}

function has_objective_front(objectives: Map<string, number> | Record<string, number>): boolean {
  return objective_entries(objectives).length > 0;
}

function sorted_val_ids(ids: Iterable<DataId>): DataId[] {
  return [...ids].sort((a, b) => String(a).localeCompare(String(b)));
}

export async function log_detailed_metrics_after_discovering_new_program({
  logger,
  gepa_state,
  new_program_idx,
  valset_evaluation,
  objective_scores,
  experiment_tracker,
  linear_pareto_front_program_idx,
  valset_size,
  val_evaluation_policy,
}: {
  logger: LoggerProtocol;
  gepa_state: ProgramStateLike;
  new_program_idx: ProgramIdx;
  valset_evaluation: ValsetEvaluationLike;
  objective_scores?: Record<string, number> | null;
  experiment_tracker: ExperimentTrackerProtocol;
  linear_pareto_front_program_idx: ProgramIdx;
  valset_size: number;
  val_evaluation_policy: EvaluationPolicy;
}): Promise<void> {
  const iteration = gepa_state.i + 1;
  const best_prog_per_agg_val_score = await val_evaluation_policy.get_best_program(gepa_state);
  const best_score_on_valset = await val_evaluation_policy.get_valset_score(best_prog_per_agg_val_score, gepa_state);
  const valset_score = await val_evaluation_policy.get_valset_score(new_program_idx, gepa_state);
  const valset_scores = valset_evaluation.scores_by_val_id;
  const coverage = valset_scores.size;

  logger.log(`Iteration ${iteration}: Valset score for new program: ${valset_score} (coverage ${coverage} / ${valset_size})`);
  logger.log(`Iteration ${iteration}: Val aggregate for new program: ${valset_score}`);
  logger.log(`Iteration ${iteration}: Individual valset scores for new program: ${String(valset_scores)}`);
  if (objective_scores != null && Object.keys(objective_scores).length > 0) {
    logger.log(`Iteration ${iteration}: Objective aggregate scores for new program: ${JSON.stringify(objective_scores)}`);
  }
  logger.log(`Iteration ${iteration}: New valset pareto front scores: ${String(gepa_state.pareto_front_valset)}`);
  if (has_objective_front(gepa_state.objective_pareto_front)) {
    logger.log(`Iteration ${iteration}: Objective pareto front scores: ${JSON.stringify(Object.fromEntries(objective_entries(gepa_state.objective_pareto_front)))}`);
  }

  const pareto_scores = [...gepa_state.pareto_front_valset.values()];
  if (pareto_scores.length === 0 || pareto_scores.some((score) => score === Number.NEGATIVE_INFINITY)) {
    throw new Error('Should have at least one valid score per validation example');
  }
  const pareto_avg = pareto_scores.reduce((sum, score) => sum + score, 0) / pareto_scores.length;

  logger.log(`Iteration ${iteration}: Valset pareto front aggregate score: ${pareto_avg}`);
  logger.log(`Iteration ${iteration}: Updated valset pareto front programs: ${String(gepa_state.program_at_pareto_front_valset)}`);
  if (Object.keys(gepa_state.program_at_pareto_front_objectives).length > 0 || gepa_state.program_at_pareto_front_objectives instanceof Map) {
    logger.log(`Iteration ${iteration}: Updated objective pareto front programs: ${String(gepa_state.program_at_pareto_front_objectives)}`);
  }
  logger.log(`Iteration ${iteration}: Best valset aggregate score so far: ${Math.max(...gepa_state.program_full_scores_val_set)}`);
  logger.log(`Iteration ${iteration}: Best program as per aggregate score on valset: ${best_prog_per_agg_val_score}`);
  logger.log(`Iteration ${iteration}: Best score on valset: ${best_score_on_valset}`);
  logger.log(`Iteration ${iteration}: Linear pareto front program index: ${linear_pareto_front_program_idx}`);
  logger.log(`Iteration ${iteration}: New program candidate index: ${new_program_idx}`);

  const metrics: Record<string, number> = {
    iteration,
    new_program_idx,
    valset_pareto_front_agg: pareto_avg,
    best_score_on_valset,
    linear_pareto_front_program_idx,
    best_program_as_per_agg_score_valset: best_prog_per_agg_val_score,
    val_evaluated_count_new_program: coverage,
    val_total_count: valset_size,
    val_program_average: valset_score,
    total_metric_calls: gepa_state.total_num_evals,
  };
  if (objective_scores != null) {
    for (const [obj_name, obj_val] of Object.entries(objective_scores)) {
      if (Number.isFinite(obj_val)) {
        metrics[`objective/${obj_name}`] = obj_val;
      }
    }
  }
  experiment_tracker.log_metrics(metrics, iteration);

  const all_val_ids = sorted_val_ids(gepa_state.pareto_front_valset.keys());
  const new_scores_dict = gepa_state.prog_candidate_val_subscores[new_program_idx] ?? new Map<DataId, number>();
  const new_parent = gepa_state.parent_program_for_candidate[new_program_idx] ?? [];
  experiment_tracker.log_table(
    'valset_scores',
    ['candidate_idx', 'parent_ids', ...all_val_ids.map(String)],
    [[new_program_idx, String(new_parent), ...all_val_ids.map((vid) => new_scores_dict.get(vid))]],
  );

  const pareto_front_rows = [...gepa_state.pareto_front_valset.entries()].map(([val_id, score]) => [
    String(val_id),
    score,
    String([...(gepa_state.program_at_pareto_front_valset.get(val_id) ?? new Set<ProgramIdx>())].sort((a, b) => a - b)),
  ]);
  if (pareto_front_rows.length > 0) {
    experiment_tracker.log_table('valset_pareto_front', ['val_id', 'best_score', 'program_ids'], pareto_front_rows);
  }

  const new_obj_scores = gepa_state.prog_candidate_objective_scores[new_program_idx] ?? {};
  if (Object.keys(new_obj_scores).length > 0) {
    const all_objectives = Object.keys(new_obj_scores).sort();
    experiment_tracker.log_table(
      'objective_scores',
      ['candidate_idx', 'parent_ids', ...all_objectives],
      [[new_program_idx, String(new_parent), ...all_objectives.map((obj) => new_obj_scores[obj])]],
    );
  }

  const obj_pareto_rows = objective_entries(gepa_state.objective_pareto_front).map(([objective, score]) => [
    objective,
    score,
    String([...objective_front_set(gepa_state.program_at_pareto_front_objectives, objective)].sort((a, b) => a - b)),
  ]);
  if (obj_pareto_rows.length > 0) {
    experiment_tracker.log_table('objective_pareto_front', ['objective', 'best_score', 'program_ids'], obj_pareto_rows);
  }
}
