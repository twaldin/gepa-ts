import type {
  BatchSampler,
  Candidate,
  CandidateProposal,
  CandidateSelector,
  DataId,
  DataLoader,
  EvaluatorOptState,
  GEPAAdapter,
  GEPACallback,
  LanguageModel,
  ProposalFn,
  ReflectionComponentSelector,
  SubsampleEvaluation,
} from './types.js';
import { SINGLE_INSTANCE_SENTINEL } from './types.js';
import { SINGLE_INSTANCE_BEST_EVALS_KEY } from './state.js';
import type { GEPAState } from './state.js';
import { InstructionProposalSignature } from './instruction_proposal.js';
import { notify_callbacks } from './callbacks.js';

export interface ProposalContext<TDataId extends DataId = DataId> {
  iteration: number;
  curr_prog_id: number;
  curr_prog: Candidate;
  curr_prog_score: number;
  subsample_ids: TDataId[];
  minibatch: unknown[];
  parent_ids: number[];
  is_seed_candidate: boolean;
}

export interface ProposalOutput<TDataId extends DataId = DataId> {
  proposal: CandidateProposal<TDataId> | null;
  total_evals: number;
  trace_data: Record<string, unknown>;
}

export class ReflectiveMutationProposer<TDataId extends DataId = DataId, TDataInst = unknown> {
  readonly trainset: DataLoader<TDataId, TDataInst>;

  private readonly logger: { log: (msg: string) => void };
  private readonly adapter: GEPAAdapter;
  private readonly candidate_selector: CandidateSelector;
  private readonly module_selector: ReflectionComponentSelector;
  private readonly batch_sampler: BatchSampler<TDataId, TDataInst>;
  private readonly perfect_score: number | null;
  private readonly skip_perfect_score: boolean;
  private readonly reflection_lm: LanguageModel | null;
  private readonly reflection_prompt_template: string | Record<string, string> | null;
  private readonly custom_candidate_proposer: ProposalFn | null;
  private readonly callbacks: GEPACallback[] | null;
  private readonly best_example_evals_k: number;
  private readonly _missing_template_warnings: Set<string>;

  constructor(opts: {
    logger: { log: (msg: string) => void };
    trainset: DataLoader<TDataId, TDataInst>;
    adapter: GEPAAdapter;
    candidate_selector: CandidateSelector;
    module_selector: ReflectionComponentSelector;
    batch_sampler: BatchSampler<TDataId, TDataInst>;
    perfect_score: number | null;
    skip_perfect_score: boolean;
    reflection_lm?: LanguageModel | null;
    reflection_prompt_template?: string | Record<string, string> | null;
    custom_candidate_proposer?: ProposalFn | null;
    callbacks?: GEPACallback[] | null;
    best_example_evals_k?: number;
  }) {
    this.logger = opts.logger;
    this.trainset = opts.trainset;
    this.adapter = opts.adapter;
    this.candidate_selector = opts.candidate_selector;
    this.module_selector = opts.module_selector;
    this.batch_sampler = opts.batch_sampler;
    this.perfect_score = opts.perfect_score ?? null;
    this.skip_perfect_score = opts.skip_perfect_score;
    this.reflection_lm = opts.reflection_lm ?? null;
    this.reflection_prompt_template = opts.reflection_prompt_template ?? null;
    this.custom_candidate_proposer = opts.custom_candidate_proposer ?? null;
    this.callbacks = opts.callbacks ?? null;
    this.best_example_evals_k = opts.best_example_evals_k ?? 30;
    this._missing_template_warnings = new Set();

    if (typeof opts.reflection_prompt_template === 'object' && opts.reflection_prompt_template !== null) {
      for (const template of Object.values(opts.reflection_prompt_template)) {
        InstructionProposalSignature.validate_prompt_template(template);
      }
    } else {
      InstructionProposalSignature.validate_prompt_template(opts.reflection_prompt_template ?? null);
    }

    if (this.skip_perfect_score && this.perfect_score === null) {
      throw new Error(
        'perfect_score must be provided when skip_perfect_score is True. ' +
          'If you do not have a perfect target score, set skip_perfect_score=False.',
      );
    }
  }

  async propose_new_texts(
    candidate: Candidate,
    reflective_dataset: Record<string, Array<Record<string, unknown>>>,
    components_to_update: string[],
  ): Promise<[Candidate, Record<string, string | Array<Record<string, unknown>>>, Record<string, string>]> {
    if (this.adapter.propose_new_texts != null) {
      const new_texts = this.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update);
      return [new_texts, {}, {}];
    }

    if (this.custom_candidate_proposer != null) {
      const new_texts = this.custom_candidate_proposer(candidate, reflective_dataset, components_to_update);
      return [new_texts, {}, {}];
    }

    if (this.reflection_lm === null) {
      throw new Error('reflection_lm must be provided when adapter.propose_new_texts is None.');
    }

    const new_texts: Candidate = {};
    const prompts: Record<string, string | Array<Record<string, unknown>>> = {};
    const raw_lm_outputs: Record<string, string> = {};

    for (const name of components_to_update) {
      if (!(name in reflective_dataset) || !reflective_dataset[name]?.length) {
        this.logger.log(`Component '${name}' is not in reflective dataset. Skipping.`);
        continue;
      }

      const base_instruction = candidate[name] ?? '';
      const dataset_with_feedback = reflective_dataset[name] ?? [];

      let prompt_template: string | null;
      if (typeof this.reflection_prompt_template === 'object' && this.reflection_prompt_template !== null) {
        prompt_template = this.reflection_prompt_template[name] ?? null;
        if (prompt_template === null && !this._missing_template_warnings.has(name)) {
          this.logger.log(`No reflection_prompt_template found for parameter '${name}'. Using default template.`);
          this._missing_template_warnings.add(name);
        }
      } else {
        prompt_template = this.reflection_prompt_template ?? null;
      }

      const { outputs: result, prompt, lm_output: raw_output } = await InstructionProposalSignature.run_with_metadata(
        this.reflection_lm as (prompt: string) => Promise<string>,
        {
          current_instruction_doc: base_instruction,
          dataset_with_feedback,
          prompt_template,
        },
      );

      new_texts[name] = result.new_instruction;
      prompts[name] = prompt;
      raw_lm_outputs[name] = raw_output;
    }

    return [new_texts, prompts, raw_lm_outputs];
  }

  prepare_proposal(state: GEPAState): ProposalContext<TDataId> {
    const i = state.i + 1;

    const curr_prog_id = this.candidate_selector.select_candidate_idx(state);
    const curr_prog = state.program_candidates[curr_prog_id] ?? {};
    const curr_prog_score = state.program_full_scores_val_set[curr_prog_id] ?? Number.NEGATIVE_INFINITY;
    this.logger.log(`Iteration ${i}: Selected program ${curr_prog_id} score: ${curr_prog_score}`);

    notify_callbacks(this.callbacks ?? undefined, 'on_candidate_selected', {
      iteration: i,
      candidate_idx: curr_prog_id,
      candidate: curr_prog,
      score: curr_prog_score,
    });

    const subsample_ids = this.batch_sampler.next_minibatch_ids(this.trainset, state) as TDataId[];
    const minibatch = this.trainset.fetch(subsample_ids) as unknown[];

    notify_callbacks(this.callbacks ?? undefined, 'on_minibatch_sampled', {
      iteration: i,
      minibatch_ids: subsample_ids,
      trainset_size: this.trainset.length,
    });

    const parent_ids_raw = state.parent_program_for_candidate[curr_prog_id] ?? [];
    const curr_parent_ids = parent_ids_raw.filter((p): p is number => p !== null);
    const is_seed_candidate = curr_prog_id === 0;

    return {
      iteration: i,
      curr_prog_id,
      curr_prog,
      curr_prog_score,
      subsample_ids,
      minibatch,
      parent_ids: curr_parent_ids,
      is_seed_candidate,
    };
  }

  private _best_evals_key_for_id(data_id: TDataId): TDataId {
    return (String(data_id) === String(SINGLE_INSTANCE_SENTINEL) ? SINGLE_INSTANCE_BEST_EVALS_KEY : data_id) as TDataId;
  }

  private _build_opt_states(state: GEPAState, data_ids: TDataId[]): EvaluatorOptState[] {
    return data_ids.map((data_id) => ({
      best_example_evals: [...(state.best_example_evals.get(this._best_evals_key_for_id(data_id)) ?? [])],
    }));
  }

  private _record_eval_batch(state: GEPAState, data_ids: TDataId[], scores: number[], side_infos?: Record<string, unknown>[]): void {
    for (let idx = 0; idx < data_ids.length; idx += 1) {
      const data_id = data_ids[idx];
      const score = scores[idx];
      if (data_id === undefined || score === undefined) {
        continue;
      }
      state.record_example_eval(data_id, score, side_infos?.[idx] ?? {}, this.best_example_evals_k);
    }
  }

  async execute_proposal(ctx: ProposalContext<TDataId>, state: GEPAState): Promise<ProposalOutput<TDataId>> {
    const i = ctx.iteration;
    const trace_data: Record<string, unknown> = {
      selected_program_candidate: ctx.curr_prog_id,
      subsample_ids: ctx.subsample_ids,
    };
    let total_evals = 0;

    notify_callbacks(this.callbacks ?? undefined, 'on_evaluation_start', {
      iteration: i,
      candidate_idx: ctx.curr_prog_id,
      batch_size: ctx.minibatch.length,
      capture_traces: true,
      parent_ids: ctx.parent_ids,
      inputs: ctx.minibatch,
      is_seed_candidate: ctx.is_seed_candidate,
    });

    const eval_curr_opt_states = this._build_opt_states(state, ctx.subsample_ids);
    const eval_curr = await this.adapter.evaluate(ctx.minibatch, ctx.curr_prog, true, eval_curr_opt_states);
    total_evals += eval_curr.num_metric_calls ?? ctx.subsample_ids.length;
    trace_data['subsample_scores'] = eval_curr.scores;
    this._record_eval_batch(state, ctx.subsample_ids, eval_curr.scores, eval_curr.side_infos);

    notify_callbacks(this.callbacks ?? undefined, 'on_evaluation_end', {
      iteration: i,
      candidate_idx: ctx.curr_prog_id,
      scores: eval_curr.scores,
      has_trajectories: !!(eval_curr.trajectories?.length),
      parent_ids: ctx.parent_ids,
      outputs: eval_curr.outputs,
      ...(eval_curr.trajectories !== undefined ? { trajectories: eval_curr.trajectories } : {}),
      ...(eval_curr.objective_scores !== undefined ? { objective_scores: eval_curr.objective_scores } : {}),
      is_seed_candidate: ctx.is_seed_candidate,
    });

    if (!eval_curr.trajectories || eval_curr.trajectories.length === 0) {
      this.logger.log(`Iteration ${i}: No trajectories captured. Skipping.`);
      notify_callbacks(this.callbacks ?? undefined, 'on_evaluation_skipped', {
        iteration: i,
        candidate_idx: ctx.curr_prog_id,
        reason: 'no_trajectories',
        scores: eval_curr.scores,
        is_seed_candidate: ctx.is_seed_candidate,
      });
      return { proposal: null, total_evals, trace_data };
    }

    if (
      this.skip_perfect_score &&
      this.perfect_score !== null &&
      eval_curr.scores.every((s) => s >= (this.perfect_score as number))
    ) {
      this.logger.log(`Iteration ${i}: All subsample scores perfect. Skipping.`);
      notify_callbacks(this.callbacks ?? undefined, 'on_evaluation_skipped', {
        iteration: i,
        candidate_idx: ctx.curr_prog_id,
        reason: 'all_scores_perfect',
        scores: eval_curr.scores,
        is_seed_candidate: ctx.is_seed_candidate,
      });
      return { proposal: null, total_evals, trace_data };
    }

    const predictor_names_to_update = this.module_selector(
      state,
      eval_curr.trajectories,
      eval_curr.scores,
      ctx.curr_prog_id,
      ctx.curr_prog,
    );

    let new_texts: Candidate;
    let prompts: Record<string, string | Array<Record<string, unknown>>>;
    let raw_lm_outputs: Record<string, string>;
    const _lm_metadata: Record<string, unknown> = {};

    try {
      const reflective_dataset = this.adapter.make_reflective_dataset(
        ctx.curr_prog,
        eval_curr,
        predictor_names_to_update,
      );

      notify_callbacks(this.callbacks ?? undefined, 'on_reflective_dataset_built', {
        iteration: i,
        candidate_idx: ctx.curr_prog_id,
        components: predictor_names_to_update,
        dataset: reflective_dataset,
      });

      notify_callbacks(this.callbacks ?? undefined, 'on_proposal_start', {
        iteration: i,
        parent_candidate: ctx.curr_prog,
        components: predictor_names_to_update,
        reflective_dataset,
      });

      [new_texts, prompts, raw_lm_outputs] = await this.propose_new_texts(
        ctx.curr_prog,
        reflective_dataset,
        predictor_names_to_update,
      );

      notify_callbacks(this.callbacks ?? undefined, 'on_proposal_end', {
        iteration: i,
        new_instructions: new_texts,
        prompts,
        raw_lm_outputs,
      });

      for (const comp of Object.keys(new_texts)) {
        _lm_metadata[`prompt:${comp}`] = prompts[comp] ?? '';
        _lm_metadata[`raw_lm_output:${comp}`] = raw_lm_outputs[comp] ?? '';
      }

      for (const [pname, text] of Object.entries(new_texts)) {
        this.logger.log(`Iteration ${i}: Proposed new text for ${pname}: ${text}`);
      }
    } catch (error) {
      this.logger.log(`Iteration ${i}: Exception during reflection/proposal: ${String(error)}`);
      return { proposal: null, total_evals, trace_data };
    }

    const new_candidate: Candidate = { ...ctx.curr_prog };
    for (const [pname, text] of Object.entries(new_texts)) {
      if (!(pname in new_candidate)) {
        throw new Error(`${pname} missing in candidate`);
      }
      new_candidate[pname] = text;
    }

    notify_callbacks(this.callbacks ?? undefined, 'on_evaluation_start', {
      iteration: i,
      candidate_idx: null,
      batch_size: ctx.minibatch.length,
      capture_traces: true,
      parent_ids: [ctx.curr_prog_id],
      inputs: ctx.minibatch,
      is_seed_candidate: false,
    });

    const eval_after_opt_states = this._build_opt_states(state, ctx.subsample_ids);
    const eval_after = await this.adapter.evaluate(ctx.minibatch, new_candidate, true, eval_after_opt_states);
    const new_scores = eval_after.scores;
    const new_outputs = eval_after.outputs;
    total_evals += eval_after.num_metric_calls ?? ctx.subsample_ids.length;
    this._record_eval_batch(state, ctx.subsample_ids, new_scores, eval_after.side_infos);

    notify_callbacks(this.callbacks ?? undefined, 'on_evaluation_end', {
      iteration: i,
      candidate_idx: null,
      scores: new_scores,
      has_trajectories: !!(eval_after.trajectories?.length),
      parent_ids: [ctx.curr_prog_id],
      outputs: new_outputs,
      ...(eval_after.trajectories !== undefined ? { trajectories: eval_after.trajectories } : {}),
      ...(eval_after.objective_scores !== undefined ? { objective_scores: eval_after.objective_scores } : {}),
      is_seed_candidate: false,
    });

    trace_data['new_subsample_scores'] = new_scores;

    const eval_before: SubsampleEvaluation = {
      scores: eval_curr.scores,
      outputs: eval_curr.outputs,
      ...(eval_curr.objective_scores !== undefined ? { objective_scores: eval_curr.objective_scores } : {}),
      ...(eval_curr.trajectories !== undefined ? { trajectories: eval_curr.trajectories } : {}),
    };
    const eval_after_sub: SubsampleEvaluation = {
      scores: new_scores,
      outputs: new_outputs,
      ...(eval_after.objective_scores !== undefined ? { objective_scores: eval_after.objective_scores } : {}),
      ...(eval_after.trajectories !== undefined ? { trajectories: eval_after.trajectories } : {}),
    };

    const proposal: CandidateProposal<TDataId> = {
      candidate: new_candidate,
      parent_program_ids: [ctx.curr_prog_id],
      subsample_indices: ctx.subsample_ids,
      subsample_scores_before: eval_curr.scores,
      subsample_scores_after: new_scores,
      eval_before,
      eval_after: eval_after_sub,
      tag: 'reflective_mutation',
      metadata: _lm_metadata,
    };

    return { proposal, total_evals, trace_data };
  }

  apply_proposal_output(output: ProposalOutput<TDataId>, state: GEPAState): void {
    state.increment_evals(output.total_evals);
  }

  async propose_output(state: GEPAState): Promise<ProposalOutput<TDataId>> {
    const ctx = this.prepare_proposal(state);
    const last_trace = state.full_program_trace[state.full_program_trace.length - 1];
    if (last_trace) {
      last_trace['selected_program_candidate'] = ctx.curr_prog_id;
      last_trace['subsample_ids'] = ctx.subsample_ids;
    }
    return this.execute_proposal(ctx, state);
  }

  async propose(state: GEPAState): Promise<CandidateProposal<TDataId> | null> {
    const output = await this.propose_output(state);
    this.apply_proposal_output(output, state);
    const last_trace = state.full_program_trace[state.full_program_trace.length - 1];
    if (last_trace) {
      Object.assign(last_trace, output.trace_data);
    }
    return output.proposal;
  }
}
