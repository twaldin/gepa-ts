import type {
  AcceptanceCriterion,
  Candidate,
  DataId,
  EvaluationPolicy,
  EvaluatorOptState,
  ExperimentTrackerProtocol,
  FrontierType,
  GEPAAdapter,
  GEPACallback,
  CandidateProposal,
  Stopper,
} from './types.js';
import { SINGLE_INSTANCE_SENTINEL } from './types.js';
import { SINGLE_INSTANCE_BEST_EVALS_KEY } from './state.js';
import { EvaluationCache, GEPAState, ValsetEvaluation, initialize_gepa_state } from './state.js';
import { ReflectiveMutationProposer } from './proposer.js';
import { MergeProposer } from './proposer/merge.js';
import type { ProposalOutput } from './proposer.js';
import { notify_callbacks } from './callbacks.js';
import { StrictImprovementAcceptance } from './acceptance.js';
import { FullEvaluationPolicy } from './eval_policy.js';
import { fetch_loader, refresh_loader } from './data_loader.js';
import { existsSync } from 'node:fs';
import { log_detailed_metrics_after_discovering_new_program } from './logging/utils.js';

const GEPA_STATE_JSON = 'gepa_state.json';

export class GEPAEngine {
  private readonly adapter: GEPAAdapter;
  private readonly run_dir: string | null;
  private readonly valset: import('./types.js').DataLoader | null;
  private readonly seed_candidate: Candidate;
  private readonly perfect_score: number | null;
  private readonly seed: number;
  readonly reflective_proposer: ReflectiveMutationProposer;
  readonly merge_proposer: MergeProposer | null;
  private readonly frontier_type: FrontierType;
  private readonly logger: { log: (msg: string) => void };
  private readonly callbacks: GEPACallback[] | null;
  private readonly track_best_outputs: boolean;
  private readonly raise_on_exception: boolean;
  private readonly stop_callback: Stopper | null;
  private readonly val_evaluation_policy: EvaluationPolicy;
  private readonly acceptance_criterion: AcceptanceCriterion;
  private readonly best_example_evals_k: number;
  private readonly initial_evaluation_cache: EvaluationCache | null;
  private readonly experiment_tracker: ExperimentTrackerProtocol;
  private _stop_requested: boolean;

  constructor(opts: {
    adapter: GEPAAdapter;
    run_dir?: string | null;
    valset: import('./types.js').DataLoader | null;
    seed_candidate: Candidate;
    perfect_score?: number | null;
    seed?: number;
    reflective_proposer: ReflectiveMutationProposer;
    merge_proposer?: MergeProposer | null;
    frontier_type?: FrontierType;
    logger: { log: (msg: string) => void };
    callbacks?: GEPACallback[] | null;
    track_best_outputs?: boolean;
    raise_on_exception?: boolean;
    stop_callback?: Stopper | null;
    val_evaluation_policy?: EvaluationPolicy | null;
    acceptance_criterion?: AcceptanceCriterion | null;
    best_example_evals_k?: number;
    evaluation_cache?: EvaluationCache | null;
    experiment_tracker: ExperimentTrackerProtocol;
  }) {
    this.adapter = opts.adapter;
    this.run_dir = opts.run_dir ?? null;
    this.valset = opts.valset;
    this.seed_candidate = opts.seed_candidate;
    this.perfect_score = opts.perfect_score ?? null;
    this.seed = opts.seed ?? 0;
    this.reflective_proposer = opts.reflective_proposer;
    this.merge_proposer = opts.merge_proposer ?? null;
    this.frontier_type = opts.frontier_type ?? 'instance';
    this.logger = opts.logger;
    this.callbacks = opts.callbacks ?? null;
    this.track_best_outputs = opts.track_best_outputs ?? false;
    this.raise_on_exception = opts.raise_on_exception ?? true;
    this.stop_callback = opts.stop_callback ?? null;
    this.val_evaluation_policy = opts.val_evaluation_policy ?? new FullEvaluationPolicy();
    this.acceptance_criterion = opts.acceptance_criterion ?? new StrictImprovementAcceptance();
    this.best_example_evals_k = opts.best_example_evals_k ?? 30;
    this.initial_evaluation_cache = opts.evaluation_cache ?? null;
    this.experiment_tracker = opts.experiment_tracker;
    this._stop_requested = false;
  }

  private _best_evals_key_for_id(val_id: DataId): DataId {
    return (String(val_id) === String(SINGLE_INSTANCE_SENTINEL) ? SINGLE_INSTANCE_BEST_EVALS_KEY : val_id) as DataId;
  }

  private _build_opt_states(state: GEPAState, val_ids: DataId[]): EvaluatorOptState[] {
    return val_ids.map((val_id) => ({
      best_example_evals: [...(state.best_example_evals.get(this._best_evals_key_for_id(val_id)) ?? [])],
    }));
  }

  private _record_eval_batch(state: GEPAState, val_ids: DataId[], scores: number[], side_infos?: Record<string, unknown>[]): void {
    for (let idx = 0; idx < val_ids.length; idx += 1) {
      const val_id = val_ids[idx];
      const score = scores[idx];
      if (val_id === undefined || score === undefined) {
        continue;
      }
      state.record_example_eval(val_id, score, side_infos?.[idx] ?? {}, this.best_example_evals_k);
    }
  }

  private async _evaluate_on_valset(program: Candidate, state: GEPAState): Promise<ValsetEvaluation> {
    const valset = this.valset;
    if (valset === null) throw new Error('valset must be provided');

    await refresh_loader(valset);
    const val_ids = await this.val_evaluation_policy.get_eval_batch(valset, state);
    const [outputs_by_val_id, scores_by_val_id, objective_scores_by_val_id, num_actual_evals] =
      await state.cached_evaluate_full(
        program,
        val_ids,
        (ids) => fetch_loader(valset, ids),
        async (batch, candidate, uncached_ids) => {
          const typed_batch = Array.isArray(batch) ? batch : [];
          const opt_states = this._build_opt_states(state, uncached_ids);
          const eval_result = await this.adapter.evaluate(typed_batch, candidate, false, opt_states);
          this._record_eval_batch(state, uncached_ids, eval_result.scores, eval_result.side_infos);
          return [eval_result.outputs, eval_result.scores, eval_result.objective_scores ?? null];
        },
      );
    state.increment_evals(num_actual_evals);

    return new ValsetEvaluation({
      outputs_by_val_id,
      scores_by_val_id,
      objective_scores_by_val_id,
    });
  }

  private async _run_full_eval_and_add(
    new_program: Candidate,
    state: GEPAState,
    parent_program_idx: number[],
  ): Promise<[number, number]> {
    const num_metric_calls_by_discovery = state.total_num_evals;
    const valset_evaluation = await this._evaluate_on_valset(new_program, state);
    state.num_full_ds_evals += 1;

    const front_before = state.get_pareto_front_mapping();
    const candidates_before = new Set<number>();
    for (const program_set of front_before.values()) {
      for (const p of program_set) candidates_before.add(p);
    }

    const new_program_idx = state.update_state_with_new_program({
      parent_program_idx,
      new_program,
      valset_evaluation,
      run_dir: this.run_dir,
      num_metric_calls_by_discovery_of_new_program: num_metric_calls_by_discovery,
    });

    const valset_score = await this.val_evaluation_policy.get_valset_score(new_program_idx, state);
    const linear_pareto_front_program_idx = await this.val_evaluation_policy.get_best_program(state);
    const is_best_program = new_program_idx === linear_pareto_front_program_idx;

    const front_after = state.get_pareto_front_mapping();
    const candidates_after = new Set<number>();
    for (const program_set of front_after.values()) {
      for (const p of program_set) candidates_after.add(p);
    }

    const new_front = [...candidates_after].sort((a, b) => a - b);
    const displaced_candidates = [...candidates_before]
      .filter((p) => !candidates_after.has(p))
      .sort((a, b) => a - b);

    notify_callbacks(this.callbacks ?? undefined, 'on_pareto_front_updated', {
      iteration: state.i + 1,
      new_front,
      displaced_candidates,
    });

    const last_trace = state.full_program_trace[state.full_program_trace.length - 1];
    if (last_trace) {
      last_trace['new_program_idx'] = new_program_idx;
      last_trace['evaluated_val_indices'] = [...valset_evaluation.scores_by_val_id.keys()].sort();
    }

    if (is_best_program) {
      this.logger.log(
        `Iteration ${state.i + 1}: Found a better program on the valset with score ${valset_score}.`,
      );
    }

    const valset = this.valset!;
    notify_callbacks(this.callbacks ?? undefined, 'on_valset_evaluated', {
      iteration: state.i + 1,
      candidate_idx: new_program_idx,
      candidate: new_program,
      scores_by_val_id: Object.fromEntries(valset_evaluation.scores_by_val_id),
      average_score: valset_score,
      num_examples_evaluated: valset_evaluation.scores_by_val_id.size,
      total_valset_size: valset.length,
      parent_ids: parent_program_idx,
      is_best_program,
      ...(valset_evaluation.outputs_by_val_id.size > 0
        ? { outputs_by_val_id: Object.fromEntries(valset_evaluation.outputs_by_val_id) }
        : {}),
    });

    await log_detailed_metrics_after_discovering_new_program({
      logger: this.logger,
      gepa_state: state,
      new_program_idx,
      valset_evaluation,
      objective_scores: state.prog_candidate_objective_scores[new_program_idx] ?? {},
      experiment_tracker: this.experiment_tracker,
      linear_pareto_front_program_idx,
      valset_size: valset.length,
      val_evaluation_policy: this.val_evaluation_policy,
    });

    const component_names = Object.keys(new_program).sort();
    this.experiment_tracker.log_table(
      'candidates',
      ['iteration', 'candidate_idx', 'parent_ids', 'valset_score', 'is_best', ...component_names.map((name) => `text:${name}`)],
      [[
        state.i + 1,
        new_program_idx,
        String(parent_program_idx),
        valset_score,
        is_best_program,
        ...component_names.map((name) => new_program[name]),
      ]],
    );

    return [new_program_idx, linear_pareto_front_program_idx];
  }

  private async _accept_reflective_proposal(
    proposal: CandidateProposal,
    iteration: number,
    state: GEPAState,
  ): Promise<boolean> {
    const old_sum = (proposal.subsample_scores_before ?? []).reduce((a, b) => a + b, 0);
    const new_sum = (proposal.subsample_scores_after ?? []).reduce((a, b) => a + b, 0);

    if (!this.acceptance_criterion.should_accept(proposal, state)) {
      this.logger.log(
        `Iteration ${iteration}: New subsample score ${new_sum} is not better than old score ${old_sum}, skipping`,
      );
      notify_callbacks(this.callbacks ?? undefined, 'on_candidate_rejected', {
        iteration,
        old_score: old_sum,
        new_score: new_sum,
        reason: `New subsample score ${new_sum} not better than old score ${old_sum}`,
      });
      this._log_proposal_lm_calls(iteration, proposal, -1);
      return false;
    }

    this.logger.log(
      `Iteration ${iteration}: New subsample score ${new_sum} is better than old score ${old_sum}. Continue to full eval and add to candidate pool.`,
    );

    const [new_idx] = await this._run_full_eval_and_add(
      proposal.candidate,
      state,
      proposal.parent_program_ids,
    );
    this._log_proposal_lm_calls(iteration, proposal, new_idx);

    notify_callbacks(this.callbacks ?? undefined, 'on_candidate_accepted', {
      iteration,
      new_candidate_idx: new_idx,
      new_score: new_sum,
      parent_ids: proposal.parent_program_ids,
    });

    return true;
  }

  private _log_proposal_lm_calls(
    iteration: number,
    proposal: CandidateProposal,
    candidate_idx: number,
  ): void {
    const metadata = proposal.metadata ?? {};
    const components = new Set<string>();
    for (const key of Object.keys(metadata)) {
      if (key.startsWith('prompt:') || key.startsWith('raw_lm_output:')) {
        components.add(key.split(':', 2)[1] ?? '');
      }
    }
    components.delete('');
    if (components.size === 0) {
      return;
    }

    const status = candidate_idx >= 0 ? 'accepted' : 'rejected';
    const subsample_before = (proposal.subsample_scores_before ?? []).reduce((sum, score) => sum + score, 0);
    const subsample_after = (proposal.subsample_scores_after ?? []).reduce((sum, score) => sum + score, 0);
    const parent_ids = JSON.stringify(proposal.parent_program_ids);
    const rows: unknown[][] = [];

    for (const component of Array.from(components).sort()) {
      const prompt = metadata[`prompt:${component}`] ?? '';
      const raw_output = metadata[`raw_lm_output:${component}`] ?? '';
      rows.push([
        iteration,
        component,
        status,
        candidate_idx,
        parent_ids,
        subsample_before,
        subsample_after,
        typeof prompt === 'string' ? prompt : JSON.stringify(prompt),
        typeof raw_output === 'string' ? raw_output : String(raw_output),
        proposal.candidate[component] ?? '',
      ]);
    }

    this.experiment_tracker.log_table(
      'proposals',
      [
        'iteration',
        'component',
        'status',
        'candidate_idx',
        'parent_ids',
        'subsample_score_before',
        'subsample_score_after',
        'prompt',
        'raw_lm_output',
        'proposed_text',
      ],
      rows,
    );
  }

  private async _process_proposal_output(
    output: ProposalOutput,
    iteration: number,
    trace_entry: Record<string, unknown>,
    state: GEPAState,
  ): Promise<boolean> {
    this.reflective_proposer.apply_proposal_output(output, state);
    Object.assign(trace_entry, output.trace_data);

    if (output.proposal === null) {
      this.logger.log(`Iteration ${iteration}: Reflective mutation did not propose a new candidate`);
      return false;
    }

    const accepted = await this._accept_reflective_proposal(output.proposal, iteration, state);
    if (accepted && this.merge_proposer !== null) {
      this.merge_proposer.last_iter_found_new_program = true;
      if (this.merge_proposer.total_merges_tested < this.merge_proposer.max_merge_invocations) {
        this.merge_proposer.merges_due += 1;
      }
    }
    return accepted;
  }

  private async _maybe_process_merge(iteration: number, state: GEPAState): Promise<boolean | null> {
    const merge_proposer = this.merge_proposer;
    if (merge_proposer === null || !merge_proposer.use_merge) {
      return null;
    }

    if (merge_proposer.merges_due <= 0 || !merge_proposer.last_iter_found_new_program) {
      merge_proposer.last_iter_found_new_program = false;
      return null;
    }

    const proposal = await merge_proposer.propose(state);
    merge_proposer.last_iter_found_new_program = false;
    if (proposal === null || proposal.tag !== 'merge') {
      return null;
    }

    const parent_sums = proposal.subsample_scores_before ?? [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
    const new_sum = (proposal.subsample_scores_after ?? []).reduce((sum, score) => sum + score, 0);

    notify_callbacks(this.callbacks ?? undefined, 'on_merge_attempted', {
      iteration,
      parent_ids: proposal.parent_program_ids,
      merged_candidate: proposal.candidate,
    });

    if (new_sum >= Math.max(...parent_sums)) {
      const [new_idx] = await this._run_full_eval_and_add(
        proposal.candidate,
        state,
        proposal.parent_program_ids,
      );
      merge_proposer.merges_due -= 1;
      merge_proposer.total_merges_tested += 1;

      notify_callbacks(this.callbacks ?? undefined, 'on_merge_accepted', {
        iteration,
        new_candidate_idx: new_idx,
        parent_ids: proposal.parent_program_ids,
      });
      notify_callbacks(this.callbacks ?? undefined, 'on_candidate_accepted', {
        iteration,
        new_candidate_idx: new_idx,
        new_score: new_sum,
        parent_ids: proposal.parent_program_ids,
      });
      return true;
    }

    this.logger.log(
      `Iteration ${iteration}: New program subsample score ${new_sum} is worse than both parents ${parent_sums}, skipping merge`,
    );
    notify_callbacks(this.callbacks ?? undefined, 'on_merge_rejected', {
      iteration,
      parent_ids: proposal.parent_program_ids,
      reason: `Merged score ${new_sum} worse than both parents ${parent_sums}`,
    });
    return false;
  }

  async run(): Promise<GEPAState> {
    const valset = this.valset;
    if (valset === null) throw new Error('valset must be provided to GEPAEngine.run()');

    await refresh_loader(this.reflective_proposer.trainset);
    await refresh_loader(valset);
    this.experiment_tracker.log_config({
      seed: this.seed,
      perfect_score: this.perfect_score,
      frontier_type: this.frontier_type,
      track_best_outputs: this.track_best_outputs,
      trainset_size: this.reflective_proposer.trainset.length,
      valset_size: valset.length,
      seed_candidate_components: Object.keys(this.seed_candidate).sort(),
      val_evaluation_policy: this.val_evaluation_policy.constructor.name,
      run_dir: this.run_dir,
    });
    notify_callbacks(this.callbacks ?? undefined, 'on_optimization_start', {
      seed_candidate: this.seed_candidate,
      trainset_size: this.reflective_proposer.trainset.length,
      valset_size: valset.length,
      config: {
        perfect_score: this.perfect_score,
        seed: this.seed,
        track_best_outputs: this.track_best_outputs,
      },
    });

    const all_ids = valset.all_ids();
    const seed_batch = await fetch_loader(valset, all_ids);
    const seed_opt_states: EvaluatorOptState[] = all_ids.map(() => ({ best_example_evals: [] }));
    const seed_eval_result = await this.adapter.evaluate(seed_batch, this.seed_candidate, false, seed_opt_states);

    const seed_outputs_by_val_id = new Map<DataId, unknown>();
    const seed_scores_by_val_id = new Map<DataId, number>();
    let seed_objective_scores_by_val_id: Map<DataId, Record<string, number>> | null = null;

    if (seed_eval_result.objective_scores) {
      seed_objective_scores_by_val_id = new Map();
    }

    for (let idx = 0; idx < all_ids.length; idx++) {
      const val_id = all_ids[idx];
      if (val_id === undefined) continue;
      seed_outputs_by_val_id.set(val_id, seed_eval_result.outputs[idx]);
      const score = seed_eval_result.scores[idx];
      if (score !== undefined) seed_scores_by_val_id.set(val_id, score);
      if (seed_objective_scores_by_val_id && seed_eval_result.objective_scores) {
        seed_objective_scores_by_val_id.set(val_id, seed_eval_result.objective_scores[idx] ?? {});
      }
    }

    const seed_valset_evaluation = new ValsetEvaluation({
      outputs_by_val_id: seed_outputs_by_val_id,
      scores_by_val_id: seed_scores_by_val_id,
      objective_scores_by_val_id: seed_objective_scores_by_val_id,
    });
    const loading_existing_state = this.run_dir !== null && existsSync(`${this.run_dir}/${GEPA_STATE_JSON}`);
    const state = initialize_gepa_state({
      run_dir: this.run_dir,
      logger: this.logger,
      seed_candidate: this.seed_candidate,
      seed_valset_evaluation,
      track_best_outputs: this.track_best_outputs,
      frontier_type: this.frontier_type,
      evaluation_cache: this.initial_evaluation_cache,
    });
    if (!loading_existing_state) {
      state.total_num_evals = seed_eval_result.num_metric_calls ?? seed_scores_by_val_id.size;
      state.evaluation_cache?.put_batch(
        this.seed_candidate,
        all_ids,
        seed_eval_result.outputs,
        seed_eval_result.scores,
        seed_eval_result.objective_scores ?? null,
      );
    }

    this._record_eval_batch(state, all_ids, seed_eval_result.scores, seed_eval_result.side_infos);

    const base_val_avg = state.get_program_average_val_subset(0)[0];
    const base_val_coverage = state.get_program_average_val_subset(0)[1];
    const pareto_scores = [...state.pareto_front_valset.values()];
    const base_pareto_avg = pareto_scores.length > 0
      ? pareto_scores.reduce((sum, score) => sum + score, 0) / pareto_scores.length
      : base_val_avg;
    this.experiment_tracker.log_metrics({
      val_program_average: base_val_avg,
      best_score_on_valset: base_val_avg,
      val_evaluated_count_new_program: base_val_coverage,
      val_total_count: valset.length,
      total_metric_calls: state.total_num_evals,
      valset_pareto_front_agg: base_pareto_avg,
      new_program_idx: 0,
      linear_pareto_front_program_idx: 0,
      best_program_as_per_agg_score_valset: 0,
    }, state.i + 1);
    this.logger.log(
      `Iteration ${state.i + 1}: Base program full valset score: ${base_val_avg} over ${valset.length} / ${valset.length} examples`,
    );

    const seed_scores = state.prog_candidate_val_subscores[0];
    notify_callbacks(this.callbacks ?? undefined, 'on_valset_evaluated', {
      iteration: 0,
      candidate_idx: 0,
      candidate: this.seed_candidate,
      scores_by_val_id: seed_scores ? Object.fromEntries(seed_scores) : {},
      average_score: base_val_avg,
      num_examples_evaluated: seed_scores?.size ?? 0,
      total_valset_size: valset.length,
      parent_ids: [],
      is_best_program: true,
      outputs_by_val_id: undefined,
    });

    state.add_budget_hook((new_total: number, delta: number) => {
      notify_callbacks(this.callbacks ?? undefined, 'on_budget_updated', {
        iteration: state.i + 1,
        metric_calls_used: new_total,
        metric_calls_delta: delta,
        metric_calls_remaining: this._get_remaining_budget(state),
      });
    });

    if (this.merge_proposer !== null) {
      this.merge_proposer.last_iter_found_new_program = false;
    }

    while (!this._should_stop(state)) {
      let iteration_started = false;
      let proposal_accepted = false;

      try {
        state.save(this.run_dir);
        notify_callbacks(this.callbacks ?? undefined, 'on_state_saved', {
          iteration: state.i + 1,
          run_dir: this.run_dir ?? undefined,
        });

        state.i += 1;
        const trace_entry: Record<string, unknown> = { i: state.i };
        state.full_program_trace.push(trace_entry);

        notify_callbacks(this.callbacks ?? undefined, 'on_iteration_start', {
          iteration: state.i + 1,
          state,
          trainset_loader: this.reflective_proposer.trainset,
        });
        iteration_started = true;

        const merge_result = await this._maybe_process_merge(state.i + 1, state);
        if (merge_result !== null) {
          proposal_accepted = merge_result;
          continue;
        }

        const output = await this.reflective_proposer.propose_output(state);
        proposal_accepted = await this._process_proposal_output(output, state.i + 1, trace_entry, state);
      } catch (error) {
        this.logger.log(`Iteration ${state.i + 1}: Exception during optimization: ${String(error)}`);
        notify_callbacks(this.callbacks ?? undefined, 'on_error', {
          iteration: state.i + 1,
          exception: error instanceof Error ? error : new Error(String(error)),
          will_continue: !this.raise_on_exception,
        });
        if (this.raise_on_exception) throw error;
      } finally {
        if (iteration_started) {
          notify_callbacks(this.callbacks ?? undefined, 'on_iteration_end', {
            iteration: state.i + 1,
            state,
            proposal_accepted,
          });
        }
      }
    }

    const best_candidate_idx = await this.val_evaluation_policy.get_best_program(state);
    state.save(this.run_dir);
    const best_candidate = state.program_candidates[best_candidate_idx] ?? {};
    const best_score = await this.val_evaluation_policy.get_valset_score(best_candidate_idx, state);
    const summary: Record<string, unknown> = {
      best_candidate_idx,
      best_valset_score: best_score,
      total_iterations: state.i,
      total_candidates: state.program_candidates.length,
    };
    for (const name of Object.keys(this.seed_candidate).sort()) {
      summary[`seed/${name}`] = this.seed_candidate[name];
      summary[`best/${name}`] = best_candidate[name];
    }
    this.experiment_tracker.log_summary(summary);
    notify_callbacks(this.callbacks ?? undefined, 'on_optimization_end', {
      best_candidate_idx,
      total_iterations: state.i,
      total_metric_calls: state.total_num_evals,
      final_state: state,
    });

    return state;
  }

  _should_stop(state: GEPAState): boolean {
    if (this._stop_requested) return true;
    if (this.stop_callback && this.stop_callback(state)) return true;
    return false;
  }

  _get_remaining_budget(state: GEPAState): number | null {
    const stop_cb = this.stop_callback;
    if (stop_cb === null) return null;

    const max_calls = (stop_cb as { max_metric_calls?: number }).max_metric_calls;
    if (typeof max_calls === 'number') {
      return Math.max(0, max_calls - state.total_num_evals);
    }

    const stoppers = (stop_cb as { stoppers?: Stopper[] }).stoppers;
    if (stoppers != null) {
      for (const stopper of stoppers) {
        const stopper_max = (stopper as { max_metric_calls?: number }).max_metric_calls;
        if (typeof stopper_max === 'number') {
          return Math.max(0, stopper_max - state.total_num_evals);
        }
      }
    }

    return null;
  }

  request_stop(): void {
    this._stop_requested = true;
  }
}
