import type {
  AcceptanceCriterion,
  Candidate,
  DataId,
  EvaluationPolicy,
  FrontierType,
  GEPAAdapter,
  GEPACallback,
  Stopper,
} from './types.js';
import { GEPAState, ValsetEvaluation, initialize_gepa_state } from './state.js';
import { ReflectiveMutationProposer } from './proposer.js';
import type { ProposalOutput } from './proposer.js';
import { notify_callbacks } from './callbacks.js';
import { StrictImprovementAcceptance } from './acceptance.js';
import { FullEvaluationPolicy } from './eval_policy.js';

export class GEPAEngine {
  private readonly adapter: GEPAAdapter;
  private readonly valset: import('./types.js').DataLoader | null;
  private readonly seed_candidate: Candidate;
  private readonly perfect_score: number | null;
  private readonly seed: number;
  readonly reflective_proposer: ReflectiveMutationProposer;
  private readonly frontier_type: FrontierType;
  private readonly logger: { log: (msg: string) => void };
  private readonly callbacks: GEPACallback[] | null;
  private readonly track_best_outputs: boolean;
  private readonly raise_on_exception: boolean;
  private readonly stop_callback: Stopper | null;
  private readonly val_evaluation_policy: EvaluationPolicy;
  private readonly acceptance_criterion: AcceptanceCriterion;
  private _stop_requested: boolean;

  constructor(opts: {
    adapter: GEPAAdapter;
    valset: import('./types.js').DataLoader | null;
    seed_candidate: Candidate;
    perfect_score?: number | null;
    seed?: number;
    reflective_proposer: ReflectiveMutationProposer;
    frontier_type?: FrontierType;
    logger: { log: (msg: string) => void };
    callbacks?: GEPACallback[] | null;
    track_best_outputs?: boolean;
    raise_on_exception?: boolean;
    stop_callback?: Stopper | null;
    val_evaluation_policy?: EvaluationPolicy | null;
    acceptance_criterion?: AcceptanceCriterion | null;
  }) {
    this.adapter = opts.adapter;
    this.valset = opts.valset;
    this.seed_candidate = opts.seed_candidate;
    this.perfect_score = opts.perfect_score ?? null;
    this.seed = opts.seed ?? 0;
    this.reflective_proposer = opts.reflective_proposer;
    this.frontier_type = opts.frontier_type ?? 'instance';
    this.logger = opts.logger;
    this.callbacks = opts.callbacks ?? null;
    this.track_best_outputs = opts.track_best_outputs ?? false;
    this.raise_on_exception = opts.raise_on_exception ?? true;
    this.stop_callback = opts.stop_callback ?? null;
    this.val_evaluation_policy = opts.val_evaluation_policy ?? new FullEvaluationPolicy();
    this.acceptance_criterion = opts.acceptance_criterion ?? new StrictImprovementAcceptance();
    this._stop_requested = false;
  }

  private async _evaluate_on_valset(program: Candidate, state: GEPAState): Promise<ValsetEvaluation> {
    const valset = this.valset;
    if (valset === null) throw new Error('valset must be provided');

    const val_ids = this.val_evaluation_policy.get_eval_batch(valset, state);
    const batch = valset.fetch(val_ids);
    const eval_result = await this.adapter.evaluate(batch, program, false);

    const outputs_by_val_id = new Map<DataId, unknown>();
    const scores_by_val_id = new Map<DataId, number>();
    let objective_scores_by_val_id: Map<DataId, Record<string, number>> | null = null;

    if (eval_result.objective_scores) {
      objective_scores_by_val_id = new Map();
    }

    for (let idx = 0; idx < val_ids.length; idx++) {
      const val_id = val_ids[idx];
      if (val_id === undefined) continue;
      outputs_by_val_id.set(val_id, eval_result.outputs[idx]);
      const score = eval_result.scores[idx];
      if (score !== undefined) scores_by_val_id.set(val_id, score);
      if (objective_scores_by_val_id && eval_result.objective_scores) {
        objective_scores_by_val_id.set(val_id, eval_result.objective_scores[idx] ?? {});
      }
    }

    state.increment_evals(val_ids.length);

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
      run_dir: null,
      num_metric_calls_by_discovery_of_new_program: num_metric_calls_by_discovery,
    });

    const valset_score = this.val_evaluation_policy.get_valset_score(new_program_idx, state);
    const linear_pareto_front_program_idx = this.val_evaluation_policy.get_best_program(state);
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

    return [new_program_idx, linear_pareto_front_program_idx];
  }

  private async _accept_reflective_proposal(
    proposal: import('./types.js').CandidateProposal,
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

    notify_callbacks(this.callbacks ?? undefined, 'on_candidate_accepted', {
      iteration,
      new_candidate_idx: new_idx,
      new_score: new_sum,
      parent_ids: proposal.parent_program_ids,
    });

    return true;
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

    return this._accept_reflective_proposal(output.proposal, iteration, state);
  }

  async run(): Promise<GEPAState> {
    const valset = this.valset;
    if (valset === null) throw new Error('valset must be provided to GEPAEngine.run()');

    const valset_evaluator = async (program: Candidate): Promise<ValsetEvaluation> => {
      const all_ids = valset.all_ids();
      const batch = valset.fetch(all_ids);
      const eval_result = await this.adapter.evaluate(batch, program, false);

      const outputs_by_val_id = new Map<DataId, unknown>();
      const scores_by_val_id = new Map<DataId, number>();
      let objective_scores_by_val_id: Map<DataId, Record<string, number>> | null = null;

      if (eval_result.objective_scores) {
        objective_scores_by_val_id = new Map();
      }

      for (let idx = 0; idx < all_ids.length; idx++) {
        const val_id = all_ids[idx];
        if (val_id === undefined) continue;
        outputs_by_val_id.set(val_id, eval_result.outputs[idx]);
        const score = eval_result.scores[idx];
        if (score !== undefined) scores_by_val_id.set(val_id, score);
        if (objective_scores_by_val_id && eval_result.objective_scores) {
          objective_scores_by_val_id.set(val_id, eval_result.objective_scores[idx] ?? {});
        }
      }

      return new ValsetEvaluation({
        outputs_by_val_id,
        scores_by_val_id,
        objective_scores_by_val_id,
      });
    };

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

    const seed_valset_evaluation = await valset_evaluator(this.seed_candidate);

    const state = initialize_gepa_state({
      run_dir: null,
      logger: this.logger,
      seed_candidate: this.seed_candidate,
      seed_valset_evaluation,
      track_best_outputs: this.track_best_outputs,
      frontier_type: this.frontier_type,
    });

    const base_val_avg = state.get_program_average_val_subset(0)[0];
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

    while (!this._should_stop(state)) {
      let iteration_started = false;
      let proposal_accepted = false;

      try {
        state.i += 1;
        const trace_entry: Record<string, unknown> = { i: state.i };
        state.full_program_trace.push(trace_entry);

        notify_callbacks(this.callbacks ?? undefined, 'on_iteration_start', {
          iteration: state.i + 1,
          state,
          trainset_loader: this.reflective_proposer.trainset,
        });
        iteration_started = true;

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

    const best_candidate_idx = this.val_evaluation_policy.get_best_program(state);
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
