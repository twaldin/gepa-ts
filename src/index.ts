declare const console: { log: (...args: unknown[]) => void };

import type {
  Candidate,
  Evaluator,
  GEPAConfig,
  GEPACallback,
  AcceptanceCriterion,
  CandidateSelector,
  EvaluationPolicy,
  ReflectionComponentSelector,
  Stopper,
} from './types.js';
import { STR_CANDIDATE_KEY, SINGLE_INSTANCE_SENTINEL } from './types.js';
import { GEPAEngine } from './engine.js';
import { ReflectiveMutationProposer } from './proposer.js';
import { GEPAResult, result_from_state } from './result.js';
import { OptimizeAnythingAdapter } from './adapter.js';
import { EvaluatorWrapper } from './evaluator_wrapper.js';
import { MaxMetricCallsStopper, CompositeStopper } from './stoppers.js';
import { ParetoCandidateSelector } from './candidate_selector.js';
import { FullEvaluationPolicy } from './eval_policy.js';
import { StrictImprovementAcceptance, ImprovementOrEqualAcceptance } from './acceptance.js';
import { RoundRobinReflectionComponentSelector, AllReflectionComponentSelector } from './component_selector.js';
import { EpochShuffledBatchSampler } from './batch_sampler.js';
import { SeededRandom } from './utils.js';
import { ensure_loader } from './data_loader.js';
import { InstructionProposalSignature } from './instruction_proposal.js';
import {
  optimize_anything_reflection_prompt_template,
  build_reflection_prompt_template,
} from './reflection_prompt.js';

export interface OptimizeAnythingOpts {
  seed_candidate: string | Candidate;
  evaluator: Evaluator;
  dataset?: unknown[] | null;
  valset?: unknown[] | null;
  objective?: string | null;
  background?: string | null;
  config?: GEPAConfig | null;
}

export async function optimize_anything(opts: OptimizeAnythingOpts): Promise<GEPAResult> {
  const { evaluator, dataset, valset, objective, background } = opts;

  if (opts.seed_candidate == null) {
    throw new Error('seed_candidate is required. Seedless mode is not supported in v1.');
  }

  const str_candidate_mode = typeof opts.seed_candidate === 'string';
  const seed_candidate: Candidate = str_candidate_mode
    ? { [STR_CANDIDATE_KEY]: opts.seed_candidate as string }
    : (opts.seed_candidate as Candidate);

  const config: GEPAConfig = opts.config ?? {};
  const engine_config = config.engine ?? {};
  const reflection_config = config.reflection ?? {};
  const tracking_config = config.tracking ?? {};

  const single_instance_mode = dataset == null && valset == null;

  const reflection_minibatch_size = reflection_config.reflection_minibatch_size ?? (single_instance_mode ? 1 : 3);

  const effective_dataset: unknown[] = single_instance_mode
    ? [SINGLE_INSTANCE_SENTINEL]
    : dataset != null
      ? dataset
      : [null];

  const wrapped_evaluator = new EvaluatorWrapper(
    evaluator,
    single_instance_mode,
    str_candidate_mode,
    engine_config.raise_on_exception ?? true,
  );

  if (engine_config.cache_evaluation === true) {
    throw new Error('cache_evaluation=true is not supported in v1. Set cache_evaluation=false or omit it.');
  }

  const active_adapter = new OptimizeAnythingAdapter({ evaluator: wrapped_evaluator });

  const train_loader = ensure_loader(effective_dataset);
  const val_loader = valset != null ? ensure_loader(valset) : train_loader;

  const reflection_lm = reflection_config.reflection_lm ?? null;
  if (typeof reflection_lm === 'string') {
    throw new Error(
      'reflection_lm must be a callable, not a string. BYO LM only (no litellm wrapper in v1).',
    );
  }

  const stop_callbacks_list: Stopper[] = [];

  if (config.stop_callbacks != null) {
    if (Array.isArray(config.stop_callbacks)) {
      stop_callbacks_list.push(...config.stop_callbacks);
    } else {
      stop_callbacks_list.push(config.stop_callbacks);
    }
  }

  if (engine_config.max_metric_calls != null) {
    stop_callbacks_list.push(new MaxMetricCallsStopper(engine_config.max_metric_calls));
  }

  if (stop_callbacks_list.length === 0) {
    throw new Error(
      'At least one stopping condition must be provided via config.engine.max_metric_calls or config.stop_callbacks.',
    );
  }

  const stop_callback: Stopper =
    stop_callbacks_list.length === 1
      ? stop_callbacks_list[0]!
      : new CompositeStopper(...stop_callbacks_list);

  const logger: { log: (msg: string) => void } = tracking_config.logger ?? { log: console.log.bind(console) };

  const seed = engine_config.seed ?? 0;
  const rng = new SeededRandom(seed);

  let candidate_selector: CandidateSelector;
  const selection_strategy = engine_config.candidate_selection_strategy ?? 'pareto';
  if (selection_strategy === 'pareto' || selection_strategy == null) {
    candidate_selector = new ParetoCandidateSelector(rng);
  } else if (typeof selection_strategy === 'string') {
    throw new Error(
      `Unknown candidate_selector strategy: ${selection_strategy}. Only 'pareto' is supported in v1.`,
    );
  } else {
    candidate_selector = selection_strategy;
  }

  let val_evaluation_policy: EvaluationPolicy;
  const vep = engine_config.val_evaluation_policy;
  if (vep == null || vep === 'full_eval') {
    val_evaluation_policy = new FullEvaluationPolicy();
  } else if (typeof vep === 'string') {
    throw new Error(`Unknown val_evaluation_policy: ${vep}`);
  } else {
    val_evaluation_policy = vep;
  }

  let acceptance_criterion: AcceptanceCriterion;
  const ac = engine_config.acceptance_criterion;
  if (ac == null || ac === 'strict_improvement') {
    acceptance_criterion = new StrictImprovementAcceptance();
  } else if (ac === 'improvement_or_equal') {
    acceptance_criterion = new ImprovementOrEqualAcceptance();
  } else if (typeof ac === 'string') {
    throw new Error(`Unknown acceptance_criterion: ${ac}`);
  } else {
    acceptance_criterion = ac;
  }

  let module_selector: ReflectionComponentSelector;
  const ms = reflection_config.module_selector;
  if (ms == null || ms === 'round_robin') {
    module_selector = new RoundRobinReflectionComponentSelector();
  } else if (ms === 'all') {
    module_selector = new AllReflectionComponentSelector();
  } else if (typeof ms === 'string') {
    throw new Error(`Unknown module_selector: ${ms}`);
  } else {
    module_selector = ms;
  }

  let batch_sampler: import('./types.js').BatchSampler;
  const bs = reflection_config.batch_sampler;
  if (bs == null || bs === 'epoch_shuffled') {
    batch_sampler = new EpochShuffledBatchSampler(reflection_minibatch_size, rng);
  } else if (typeof bs === 'string') {
    throw new Error(`Unknown batch_sampler: ${bs}`);
  } else {
    batch_sampler = bs;
  }

  const user_provided_custom_template =
    reflection_config.reflection_prompt_template != null &&
    reflection_config.reflection_prompt_template !== optimize_anything_reflection_prompt_template;
  const user_provided_objective_or_background = !!(objective || background);

  if (user_provided_custom_template && user_provided_objective_or_background) {
    throw new Error(
      "Cannot specify both 'objective'/'background' and a custom 'config.reflection.reflection_prompt_template'.",
    );
  }

  let reflection_prompt_template: string | Record<string, string> | null;
  if (user_provided_objective_or_background) {
    reflection_prompt_template = build_reflection_prompt_template({
      ...(objective ? { objective } : {}),
      ...(background ? { background } : {}),
    });
  } else {
    reflection_prompt_template = reflection_config.reflection_prompt_template ?? optimize_anything_reflection_prompt_template;
  }

  if (typeof reflection_prompt_template === 'string') {
    InstructionProposalSignature.validate_prompt_template(reflection_prompt_template);
  } else if (typeof reflection_prompt_template === 'object' && reflection_prompt_template !== null) {
    for (const [param_name, template] of Object.entries(reflection_prompt_template)) {
      try {
        InstructionProposalSignature.validate_prompt_template(template);
      } catch (e) {
        throw new Error(`Invalid reflection_prompt_template for parameter '${param_name}': ${String(e)}`);
      }
    }
  }

  const proposer = new ReflectiveMutationProposer({
    logger,
    trainset: train_loader,
    adapter: active_adapter,
    candidate_selector,
    module_selector,
    batch_sampler,
    perfect_score: reflection_config.perfect_score ?? null,
    skip_perfect_score: reflection_config.skip_perfect_score ?? false,
    reflection_lm: reflection_lm ?? null,
    reflection_prompt_template,
    custom_candidate_proposer: reflection_config.custom_candidate_proposer ?? null,
    callbacks: config.callbacks ?? null,
  });

  const engine = new GEPAEngine({
    adapter: active_adapter,
    valset: val_loader,
    seed_candidate,
    perfect_score: reflection_config.perfect_score ?? null,
    seed,
    reflective_proposer: proposer,
    frontier_type: engine_config.frontier_type ?? 'instance',
    logger,
    callbacks: config.callbacks ?? null,
    track_best_outputs: engine_config.track_best_outputs ?? false,
    raise_on_exception: engine_config.raise_on_exception ?? true,
    stop_callback,
    val_evaluation_policy,
    acceptance_criterion,
  });

  const state = await engine.run();

  return result_from_state(state, {
    str_candidate_key: str_candidate_mode ? STR_CANDIDATE_KEY : null,
    run_dir: engine_config.run_dir ?? null,
    seed,
  });
}

export { GEPAEngine } from './engine.js';
export { OptimizeAnythingAdapter } from './adapter.js';
export { ReflectiveMutationProposer } from './proposer.js';
export { MaxMetricCallsStopper, CompositeStopper } from './stoppers.js';
export { STR_CANDIDATE_KEY, SINGLE_INSTANCE_SENTINEL } from './types.js';
export type {
  GEPAConfig,
  EngineConfig,
  ReflectionConfig,
  TrackingConfig,
  MergeConfig,
  RefinerConfig,
  Candidate,
  SideInfo,
  EvaluationBatch,
  GEPAAdapter,
  Evaluator,
  LanguageModel,
  GEPACallback,
  OptimizationStartEvent,
  OptimizationEndEvent,
  IterationStartEvent,
  IterationEndEvent,
  CandidateSelectedEvent,
  MinibatchSampledEvent,
  EvaluationStartEvent,
  EvaluationEndEvent,
  EvaluationSkippedEvent,
  ReflectiveDatasetBuiltEvent,
  ProposalStartEvent,
  ProposalEndEvent,
  CandidateAcceptedEvent,
  CandidateRejectedEvent,
  MergeAttemptedEvent,
  MergeAcceptedEvent,
  MergeRejectedEvent,
  ParetoFrontUpdatedEvent,
  ValsetEvaluatedEvent,
  StateSavedEvent,
  BudgetUpdatedEvent,
  ErrorEvent,
} from './types.js';
export type { GEPAResult } from './result.js';
