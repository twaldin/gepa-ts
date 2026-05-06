export type RolloutOutput = unknown;
export type Trajectory = unknown;
export type DataInst = unknown;
export type Candidate = Record<string, string>;
export type SideInfo = Record<string, unknown>;
export type DataId = string | number;
export type ProgramIdx = number;

export const STR_CANDIDATE_KEY = "current_candidate" as const;
export const SINGLE_INSTANCE_SENTINEL = Symbol.for("gepa.single_instance_sentinel");

export type ChatMessage = {
  role: string;
  content: string | unknown[];
};

export type EvalResult = number | [number, SideInfo];
export interface EvaluatorOptState {
  best_example_evals: Array<{ score: number; side_info: Record<string, unknown> }>;
}

export type Evaluator = {
  (candidate: string | Candidate): EvalResult | Promise<EvalResult>;
  (candidate: string | Candidate, ctx: { example?: unknown; opt_state?: EvaluatorOptState }): EvalResult | Promise<EvalResult>;
};

export interface EvaluationBatch<TTrajectory = Trajectory, TRolloutOutput = RolloutOutput> {
  outputs: TRolloutOutput[];
  scores: number[];
  trajectories?: TTrajectory[];
  side_infos?: SideInfo[];
  objective_scores?: Array<Record<string, number>>;
  num_metric_calls?: number;
}

export type ProposalFn = (
  candidate: Candidate,
  reflective_dataset: Record<string, Array<Record<string, unknown>>>,
  components_to_update: string[],
) => Candidate;

export interface GEPAAdapter<TDataInst = DataInst, TTrajectory = Trajectory, TRolloutOutput = RolloutOutput> {
  evaluate(
    batch: TDataInst[],
    candidate: Candidate,
    capture_traces?: boolean,
    opt_states?: Array<EvaluatorOptState | undefined>,
  ): Promise<EvaluationBatch<TTrajectory, TRolloutOutput>>;
  make_reflective_dataset(
    candidate: Candidate,
    eval_batch: EvaluationBatch<TTrajectory, TRolloutOutput>,
    components_to_update: string[],
  ): Record<string, Array<Record<string, unknown>>>;
  propose_new_texts?: ProposalFn | null;
}

export interface DataLoader<TDataId extends DataId = DataId, TDataInst = DataInst> {
  all_ids(): TDataId[];
  fetch(ids: TDataId[]): TDataInst[];
  readonly length: number;
}

export interface SubsampleEvaluation {
  scores: number[];
  outputs?: unknown[];
  objective_scores?: Array<Record<string, number>>;
  trajectories?: unknown[];
}

export interface CandidateProposal<TDataId extends DataId = DataId> {
  candidate: Candidate;
  parent_program_ids: number[];
  subsample_indices?: TDataId[];
  subsample_scores_before?: number[];
  subsample_scores_after?: number[];
  eval_before?: SubsampleEvaluation;
  eval_after?: SubsampleEvaluation;
  tag?: string;
  metadata?: Record<string, unknown>;
}

export interface GEPAStateLike {
  total_num_evals: number;
}

export type Stopper = (state: GEPAStateLike) => boolean;

export interface ProposeNewCandidate<TDataId extends DataId = DataId> {
  propose(state: GEPAStateLike): CandidateProposal<TDataId> | null;
}

export interface CandidateSelector {
  select_candidate_idx(state: GEPAStateLike): number;
}

export type ReflectionComponentSelector = (
  state: GEPAStateLike,
  trajectories: Trajectory[],
  subsample_scores: number[],
  candidate_idx: number,
  candidate: Candidate,
) => string[];

export type LanguageModel = (prompt: string | ChatMessage[]) => Promise<string>;

export interface Signature {
  prompt_template: string;
  input_keys: string[];
  output_keys: string[];
}

export interface AcceptanceCriterion {
  should_accept(proposal: CandidateProposal, state: GEPAStateLike): boolean;
}

export interface EvaluationPolicy<TDataId extends DataId = DataId, TDataInst = DataInst> {
  get_eval_batch(loader: DataLoader<TDataId, TDataInst>, state: GEPAStateLike, target_program_idx?: ProgramIdx): TDataId[];
  get_best_program(state: GEPAStateLike): ProgramIdx;
  get_valset_score(program_idx: ProgramIdx, state: GEPAStateLike): number;
}

export interface BatchSampler<TDataId extends DataId = DataId, TDataInst = DataInst> {
  next_minibatch_ids(loader: DataLoader<TDataId, TDataInst>, state: GEPAStateLike): TDataId[];
}

export type FrontierType = "instance";

export interface EngineConfig {
  run_dir?: string;
  seed?: number;
  display_progress_bar?: boolean;
  raise_on_exception?: boolean;
  use_cloudpickle?: boolean;
  track_best_outputs?: boolean;
  max_metric_calls?: number;
  max_candidate_proposals?: number;
  max_reflection_cost?: number;
  val_evaluation_policy?: EvaluationPolicy | "full_eval";
  candidate_selection_strategy?: CandidateSelector | "pareto" | "current_best" | "epsilon_greedy" | "top_k_pareto";
  frontier_type?: FrontierType;
  acceptance_criterion?: AcceptanceCriterion | "strict_improvement" | "improvement_or_equal";
  parallel?: boolean;
  max_workers?: number;
  num_parallel_proposals?: number | "auto";
  cache_evaluation?: boolean;
  cache_evaluation_storage?: "auto" | string;
  best_example_evals_k?: number;
  capture_stdio?: boolean;
}

export interface ReflectionConfig {
  skip_perfect_score?: boolean;
  perfect_score?: number;
  batch_sampler?: BatchSampler | "epoch_shuffled";
  reflection_minibatch_size?: number;
  module_selector?: ReflectionComponentSelector | "round_robin" | "all";
  reflection_lm?: LanguageModel | string | null;
  reflection_lm_kwargs?: Record<string, unknown>;
  reflection_prompt_template?: string | Record<string, string>;
  custom_candidate_proposer?: ProposalFn | null;
}

export interface MergeConfig {
  max_merge_invocations?: number;
  merge_val_overlap_floor?: number;
}

export interface RefinerConfig {
  refiner_lm?: LanguageModel | string | null;
  max_refinements?: number;
}

export interface TrackingConfig {
  logger?: LoggerProtocol;
  use_wandb?: boolean;
  wandb_api_key?: string;
  wandb_init_kwargs?: Record<string, unknown>;
  wandb_attach_existing?: boolean;
  wandb_step_metric?: string;
  use_mlflow?: boolean;
  mlflow_tracking_uri?: string;
  mlflow_experiment_name?: string;
  mlflow_attach_existing?: boolean;
  key_prefix?: string;
}

export interface GEPAConfig {
  engine?: EngineConfig;
  reflection?: ReflectionConfig;
  tracking?: TrackingConfig;
  merge?: MergeConfig | null;
  refiner?: RefinerConfig | null;
  stop_callbacks?: Stopper | Stopper[] | null;
  callbacks?: GEPACallback[] | null;
}

export interface LoggerProtocol {
  log(message: string): void;
  log_metrics(metrics: Record<string, number>, step?: number): void;
  close?(): void;
}

export interface OptimizationStartEvent {
  seed_candidate: Candidate;
  trainset_size: number;
  valset_size: number;
  config: Record<string, unknown>;
}
export interface OptimizationEndEvent {
  best_candidate_idx: number;
  total_iterations: number;
  total_metric_calls: number;
  final_state: GEPAStateLike;
}
export interface IterationStartEvent {
  iteration: number;
  state: GEPAStateLike;
  trainset_loader: DataLoader;
}
export interface IterationEndEvent {
  iteration: number;
  state: GEPAStateLike;
  proposal_accepted: boolean;
}
export interface CandidateSelectedEvent {
  iteration: number;
  candidate_idx: number;
  candidate: Candidate;
  score: number;
}
export interface MinibatchSampledEvent {
  iteration: number;
  minibatch_ids: unknown[];
  trainset_size: number;
}
export interface EvaluationStartEvent {
  iteration: number;
  candidate_idx: number | null;
  batch_size: number;
  capture_traces: boolean;
  parent_ids: ProgramIdx[];
  inputs: unknown[];
  is_seed_candidate: boolean;
}
export interface EvaluationEndEvent {
  iteration: number;
  candidate_idx: number | null;
  scores: number[];
  has_trajectories: boolean;
  parent_ids: ProgramIdx[];
  outputs: unknown[];
  trajectories?: unknown[];
  objective_scores?: Array<Record<string, number>>;
  is_seed_candidate: boolean;
}
export interface EvaluationSkippedEvent {
  iteration: number;
  candidate_idx: number;
  reason: string;
  scores?: number[];
  is_seed_candidate: boolean;
}
export interface ReflectiveDatasetBuiltEvent {
  iteration: number;
  candidate_idx: number;
  components: string[];
  dataset: Record<string, Array<Record<string, unknown>>>;
}
export interface ProposalStartEvent {
  iteration: number;
  parent_candidate: Candidate;
  components: string[];
  reflective_dataset: Record<string, Array<Record<string, unknown>>>;
}
export interface ProposalEndEvent {
  iteration: number;
  new_instructions: Candidate;
  prompts: Record<string, string | Array<Record<string, unknown>>>;
  raw_lm_outputs: Record<string, string>;
}
export interface CandidateAcceptedEvent {
  iteration: number;
  new_candidate_idx: number;
  new_score: number;
  parent_ids: ProgramIdx[];
}
export interface CandidateRejectedEvent {
  iteration: number;
  old_score: number;
  new_score: number;
  reason: string;
}
export interface MergeAttemptedEvent {
  iteration: number;
  parent_ids: ProgramIdx[];
  merged_candidate: Candidate;
}
export interface MergeAcceptedEvent {
  iteration: number;
  new_candidate_idx: number;
  parent_ids: ProgramIdx[];
}
export interface MergeRejectedEvent {
  iteration: number;
  parent_ids: ProgramIdx[];
  reason: string;
}
export interface ParetoFrontUpdatedEvent {
  iteration: number;
  new_front: number[];
  displaced_candidates: number[];
}
export interface ValsetEvaluatedEvent {
  iteration: number;
  candidate_idx: number;
  candidate: Candidate;
  scores_by_val_id: Record<DataId, number>;
  average_score: number;
  num_examples_evaluated: number;
  total_valset_size: number;
  parent_ids: ProgramIdx[];
  is_best_program: boolean;
  outputs_by_val_id?: Record<DataId, unknown>;
}
export interface StateSavedEvent {
  iteration: number;
  run_dir?: string;
}
export interface BudgetUpdatedEvent {
  iteration: number;
  metric_calls_used: number;
  metric_calls_delta: number;
  metric_calls_remaining?: number;
}
export interface ErrorEvent {
  iteration: number;
  exception: Error;
  will_continue: boolean;
}

export interface GEPACallback {
  on_optimization_start?(event: OptimizationStartEvent): void;
  on_optimization_end?(event: OptimizationEndEvent): void;
  on_iteration_start?(event: IterationStartEvent): void;
  on_iteration_end?(event: IterationEndEvent): void;
  on_candidate_selected?(event: CandidateSelectedEvent): void;
  on_minibatch_sampled?(event: MinibatchSampledEvent): void;
  on_evaluation_start?(event: EvaluationStartEvent): void;
  on_evaluation_end?(event: EvaluationEndEvent): void;
  on_evaluation_skipped?(event: EvaluationSkippedEvent): void;
  on_valset_evaluated?(event: ValsetEvaluatedEvent): void;
  on_reflective_dataset_built?(event: ReflectiveDatasetBuiltEvent): void;
  on_proposal_start?(event: ProposalStartEvent): void;
  on_proposal_end?(event: ProposalEndEvent): void;
  on_candidate_accepted?(event: CandidateAcceptedEvent): void;
  on_candidate_rejected?(event: CandidateRejectedEvent): void;
  on_merge_attempted?(event: MergeAttemptedEvent): void;
  on_merge_accepted?(event: MergeAcceptedEvent): void;
  on_merge_rejected?(event: MergeRejectedEvent): void;
  on_pareto_front_updated?(event: ParetoFrontUpdatedEvent): void;
  on_state_saved?(event: StateSavedEvent): void;
  on_budget_updated?(event: BudgetUpdatedEvent): void;
  on_error?(event: ErrorEvent): void;
}
