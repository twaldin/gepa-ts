declare const console: { log: (...args: unknown[]) => void };

import { mkdirSync, writeFileSync } from 'node:fs';
import type {
  Candidate,
  Evaluator,
  GEPAConfig,
  GEPACallback,
  AcceptanceCriterion,
  CandidateSelector,
  EvaluationPolicy,
  GEPAAdapter,
  ReflectionComponentSelector,
  Stopper,
  DataLoader,
  DataId,
} from './types.js';
import { STR_CANDIDATE_KEY, SINGLE_INSTANCE_SENTINEL } from './types.js';
import { GEPAEngine } from './engine.js';
import { ReflectiveMutationProposer } from './proposer.js';
import { MergeProposer } from './proposer/merge.js';
import { GEPAResult, result_from_state } from './result.js';
import { EvaluationCache } from './state.js';
import { OptimizeAnythingAdapter } from './adapter.js';
import { EvaluatorWrapper } from './evaluator_wrapper.js';
import {
  MaxMetricCallsStopper,
  CompositeStopper,
  FileStopper,
  MaxCandidateProposalsStopper,
  MaxReflectionCostStopper,
} from './stoppers.js';
import {
  ParetoCandidateSelector,
  CurrentBestCandidateSelector,
  EpsilonGreedyCandidateSelector,
  TopKParetoCandidateSelector,
} from './candidate_selector.js';
import { FullEvaluationPolicy } from './eval_policy.js';
import { StrictImprovementAcceptance, ImprovementOrEqualAcceptance } from './acceptance.js';
import { RoundRobinReflectionComponentSelector, AllReflectionComponentSelector } from './component_selector.js';
import { EpochShuffledBatchSampler } from './batch_sampler.js';
import { SeededRandom } from './utils.js';
import { ensure_loader, fetch_loader, refresh_loader } from './data_loader.js';
import { ensure_tracking_lm, make_litellm_lm, type LMCompletionHook, type LMOptions } from './lm.js';
import { InstructionProposalSignature } from './instruction_proposal.js';
import {
  optimize_anything_reflection_prompt_template,
  build_reflection_prompt_template,
} from './reflection_prompt.js';
import { create_experiment_tracker } from './logging/index.js';
export { LogContext, oa_log, getLogContext, setLogContext, getLogContextOrThrow } from './log_context.js';

export const DEFAULT_REFINER_PROMPT = `You are a refinement agent improving candidates in an optimization loop.

## What We're Optimizing For
The overall optimization objective is:
{objective}

This tells you what "better" means - use it to guide your improvements.

## Domain Knowledge
{background}

## Your Task
Given a candidate and its evaluation feedback:
1. Understand why it scored the way it did
2. Fix any errors (errors = zero score)
3. Make improvements that move toward the objective
4. Return the complete improved candidate
`;

export interface OptimizeAnythingOpts {
  seed_candidate?: string | Candidate | null;
  evaluator?: Evaluator;
  adapter?: GEPAAdapter;
  dataset?: unknown[] | DataLoader<DataId, unknown> | null;
  valset?: unknown[] | DataLoader<DataId, unknown> | null;
  objective?: string | null;
  background?: string | null;
  config?: GEPAConfig | null;
}

function has_completion_hook(options: Record<string, unknown> | undefined): options is LMOptions & {
  completion: LMCompletionHook;
} {
  return typeof options?.completion === 'function';
}

export function _build_seed_generation_prompt({
  objective,
  background = null,
  dataset = null,
}: {
  objective: string;
  background?: string | null;
  dataset?: unknown[] | null;
}): string {
  const sections: string[] = [];
  const render_example = (value: unknown): string => {
    if (typeof value === 'string') return value;
    if (value === null || value === undefined) return String(value);
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  };

  sections.push(
    'You are an expert assistant. Your task is to generate an initial candidate ' +
      'that will be iteratively refined by an optimization system.',
  );

  sections.push(`\n## Goal\n\n${objective}`);

  if (background) {
    sections.push(`\n## Domain Context & Constraints\n\n${background}`);
  }

  if (dataset !== null) {
    const examples = dataset.slice(0, 3);
    const example_lines = examples.map((ex, i) => `- Example ${i + 1}: ${render_example(ex)}`);
    sections.push(
      '\n## Sample Inputs\n\n' +
        'The candidate will be evaluated on inputs like these:\n\n' +
        example_lines.join('\n'),
    );
  }

  sections.push(
    '\n## Output Format\n\n' +
      'Generate a strong initial candidate based on the goal above.\n' +
      'Provide ONLY the candidate within ``` blocks. ' +
      'Do not include explanations or commentary outside the ``` blocks.',
  );

  return sections.join('\n');
}

export async function _generate_seed_candidate({
  lm,
  objective,
  background = null,
  dataset = null,
  logger = null,
}: {
  lm: (prompt: string) => Promise<string>;
  objective: string;
  background?: string | null;
  dataset?: unknown[] | null;
  logger?: { log: (msg: string) => void } | null;
}): Promise<Candidate> {
  const prompt = _build_seed_generation_prompt({ objective, background, dataset });
  logger?.log('Generating initial seed candidate via LLM...');
  const lm_output = await lm(prompt);
  const generated_text = InstructionProposalSignature.output_extractor(lm_output).new_instruction;
  logger?.log(`Generated seed candidate (${generated_text.length} chars)`);
  return { [STR_CANDIDATE_KEY]: generated_text };
}

export async function optimize_anything(opts: OptimizeAnythingOpts): Promise<GEPAResult> {
  const { evaluator, adapter, dataset, valset, objective, background } = opts;
  const provided_seed_candidate = opts.seed_candidate;

  const config: GEPAConfig = opts.config ?? {};
  const engine_config = config.engine ?? {};
  const reflection_config = config.reflection ?? {};
  const tracking_config = config.tracking ?? {};
  const raw_reflection_lm = reflection_config.reflection_lm;
  const reflection_lm_kwargs = reflection_config.reflection_lm_kwargs;
  if (typeof raw_reflection_lm === 'string' && !has_completion_hook(reflection_lm_kwargs)) {
    throw new Error(
      'LM requires an injected completion hook in the zero-dependency TypeScript build. ' +
        'Pass config.reflection.reflection_lm_kwargs.completion when reflection_lm is a model name.',
    );
  }
  const reflection_lm = typeof raw_reflection_lm === 'string'
    ? make_litellm_lm(raw_reflection_lm, reflection_lm_kwargs)
    : typeof raw_reflection_lm === 'function'
      ? ensure_tracking_lm(raw_reflection_lm)
      : raw_reflection_lm;
  let dataset_for_prompt: unknown[] | null = null;
  if (Array.isArray(dataset)) {
    dataset_for_prompt = dataset;
  } else if (dataset != null) {
    await refresh_loader(dataset);
    dataset_for_prompt = (await fetch_loader(dataset, dataset.all_ids())).slice(0, 3);
  }

  const needs_seed_generation = provided_seed_candidate == null;
  const str_candidate_mode = needs_seed_generation || typeof provided_seed_candidate === 'string';
  let seed_candidate: Candidate;
  if (needs_seed_generation) {
    if (typeof objective !== 'string' || objective.trim() === '') {
      throw new Error("'objective' is required when seed_candidate is None.");
    }
    if (typeof reflection_lm !== 'function') {
      throw new Error('reflection_lm is required when seed_candidate is None.');
    }
    seed_candidate = await _generate_seed_candidate({
      lm: reflection_lm,
      objective,
      background: background ?? null,
      dataset: dataset_for_prompt,
      logger: tracking_config.logger ?? { log: console.log.bind(console) },
    });
  } else {
    seed_candidate = typeof provided_seed_candidate === 'string'
      ? { [STR_CANDIDATE_KEY]: provided_seed_candidate }
      : provided_seed_candidate;
  }

  const single_instance_mode = dataset == null && valset == null;

  const reflection_minibatch_size = reflection_config.reflection_minibatch_size ?? (single_instance_mode ? 1 : 3);

  const effective_dataset: unknown[] | DataLoader<DataId, unknown> = single_instance_mode
    ? [SINGLE_INSTANCE_SENTINEL]
    : dataset != null
      ? dataset
      : [null];

  if (adapter === undefined && evaluator === undefined) {
    throw new Error("optimize_anything requires either 'evaluator' or 'adapter'.");
  }

  if (config.refiner != null && seed_candidate['refiner_prompt'] === undefined) {
    seed_candidate = {
      ...seed_candidate,
      refiner_prompt: DEFAULT_REFINER_PROMPT
        .replace('{objective}', objective || 'Maximize the score')
        .replace('{background}', background || 'No additional background provided.'),
    };
  }

  const active_refiner_config = config.refiner == null
    ? null
    : {
        ...config.refiner,
        refiner_lm: typeof config.refiner.refiner_lm === 'string'
          ? make_litellm_lm(config.refiner.refiner_lm)
          : typeof config.refiner.refiner_lm === 'function'
            ? config.refiner.refiner_lm
            : typeof reflection_lm === 'function'
              ? reflection_lm
              : null,
      };

  let evaluation_cache: EvaluationCache | null = null;
  if (engine_config.cache_evaluation === true) {
    const cache_storage = engine_config.cache_evaluation_storage ?? 'auto';
    const use_disk_cache = cache_storage === 'disk' || (cache_storage === 'auto' && engine_config.run_dir != null);
    if (cache_storage === 'disk' && engine_config.run_dir == null) {
      throw new Error("cache_evaluation_storage='disk' requires run_dir in EngineConfig");
    }
    evaluation_cache = new EvaluationCache();
    if (use_disk_cache && engine_config.run_dir != null) {
      const cache_dir = `${engine_config.run_dir}/fitness_cache`;
      mkdirSync(cache_dir, { recursive: true });
      writeFileSync(`${cache_dir}/gepa-ts-placeholder.pkl`, '');
    }
  }

  const active_adapter = adapter ?? new OptimizeAnythingAdapter({
    evaluator: new EvaluatorWrapper(
      evaluator!,
      single_instance_mode,
      engine_config.capture_stdio ?? false,
      str_candidate_mode,
      engine_config.raise_on_exception ?? true,
    ),
    refiner_config: active_refiner_config,
  });

  const train_loader = ensure_loader(effective_dataset);
  const val_loader = valset != null ? ensure_loader(valset) : train_loader;

  if (typeof reflection_lm !== 'function') {
    throw new Error(
      'optimize_anything: config.reflection.reflection_lm must be a function ' +
        '(prompt: string) => Promise<string>. v1 is BYO LM — no built-in adapter.',
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

  if (engine_config.run_dir != null) {
    stop_callbacks_list.push(new FileStopper(`${engine_config.run_dir}/gepa.stop`));
  }

  if (engine_config.max_metric_calls != null) {
    stop_callbacks_list.push(new MaxMetricCallsStopper(engine_config.max_metric_calls));
  }

  if (engine_config.max_candidate_proposals != null) {
    stop_callbacks_list.push(new MaxCandidateProposalsStopper(engine_config.max_candidate_proposals));
  }

  if (engine_config.max_reflection_cost != null) {
    stop_callbacks_list.push(new MaxReflectionCostStopper(engine_config.max_reflection_cost, reflection_lm));
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
  const experiment_tracker = tracking_config.experiment_tracker ?? create_experiment_tracker({
    use_wandb: tracking_config.use_wandb ?? false,
    wandb_api_key: tracking_config.wandb_api_key ?? null,
    wandb_init_kwargs: tracking_config.wandb_init_kwargs ?? null,
    wandb_attach_existing: tracking_config.wandb_attach_existing ?? false,
    wandb_step_metric: tracking_config.wandb_step_metric ?? null,
    use_mlflow: tracking_config.use_mlflow ?? false,
    mlflow_tracking_uri: tracking_config.mlflow_tracking_uri ?? null,
    mlflow_experiment_name: tracking_config.mlflow_experiment_name ?? null,
    mlflow_attach_existing: tracking_config.mlflow_attach_existing ?? false,
    key_prefix: tracking_config.key_prefix ?? '',
  });

  const seed = engine_config.seed ?? 0;
  const rng = new SeededRandom(seed);

  let candidate_selector: CandidateSelector;
  const selection_strategy = engine_config.candidate_selection_strategy ?? 'pareto';
  if (selection_strategy === 'pareto' || selection_strategy == null) {
    candidate_selector = new ParetoCandidateSelector(rng);
  } else if (selection_strategy === 'current_best') {
    candidate_selector = new CurrentBestCandidateSelector();
  } else if (selection_strategy === 'epsilon_greedy') {
    candidate_selector = new EpsilonGreedyCandidateSelector(0.1, rng);
  } else if (selection_strategy === 'top_k_pareto') {
    candidate_selector = new TopKParetoCandidateSelector(5, rng);
  } else if (typeof selection_strategy === 'string') {
    throw new Error(
      `Unknown candidate_selector strategy: ${selection_strategy}. Supported strategies: 'pareto', 'current_best', 'epsilon_greedy', 'top_k_pareto'.`,
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
    if (reflection_config.reflection_minibatch_size !== undefined) {
      throw new Error("reflection_minibatch_size only accepted if batch_sampler is 'epoch_shuffled'.");
    }
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

  const best_example_evals_k = engine_config.best_example_evals_k ?? 30;

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
    best_example_evals_k,
  });

  const merge_config = config.merge ?? null;
  const merge_proposer = merge_config === null
    ? null
    : new MergeProposer({
        logger,
        valset: val_loader,
        evaluator: async (batch, candidate) => {
          const eval_out = await active_adapter.evaluate(batch, candidate, false);
          return [eval_out.outputs, eval_out.scores, eval_out.objective_scores ?? null];
        },
        use_merge: true,
        max_merge_invocations: merge_config.max_merge_invocations ?? 5,
        rng,
        val_overlap_floor: merge_config.merge_val_overlap_floor ?? 5,
        callbacks: config.callbacks ?? null,
      });

  const engine = new GEPAEngine({
    adapter: active_adapter,
    run_dir: engine_config.run_dir ?? null,
    valset: val_loader,
    seed_candidate,
    perfect_score: reflection_config.perfect_score ?? null,
    seed,
    reflective_proposer: proposer,
    merge_proposer,
    frontier_type: engine_config.frontier_type ?? 'instance',
    logger,
    callbacks: config.callbacks ?? null,
    track_best_outputs: engine_config.track_best_outputs ?? false,
    raise_on_exception: engine_config.raise_on_exception ?? true,
    stop_callback,
    val_evaluation_policy,
    acceptance_criterion,
    best_example_evals_k,
    evaluation_cache,
    experiment_tracker,
  });

  const state = await engine.run();

  return result_from_state(state, {
    str_candidate_key: str_candidate_mode ? STR_CANDIDATE_KEY : null,
    run_dir: engine_config.run_dir ?? null,
    seed,
  });
}

export { GEPAEngine } from './engine.js';
export { EvaluationCache, ValsetEvaluation, initialize_gepa_state } from './state.js';
export { OptimizeAnythingAdapter } from './adapter.js';
export { ListDataLoader, StagedDataLoader, ensure_loader } from './data_loader.js';
export { ReflectiveMutationProposer } from './proposer.js';
export { MergeProposer } from './proposer/merge.js';
export {
  ParetoCandidateSelector,
  CurrentBestCandidateSelector,
  EpsilonGreedyCandidateSelector,
  TopKParetoCandidateSelector,
} from './candidate_selector.js';
export { FullEvaluationPolicy, RoundRobinSampleEvaluationPolicy } from './eval_policy.js';
export { MaxMetricCallsStopper, CompositeStopper } from './stoppers.js';
export { FileStopper } from './stoppers.js';
export { MaxCandidateProposalsStopper } from './stoppers.js';
export { MaxReflectionCostStopper } from './stoppers.js';
export { TimeoutStopCondition } from './stoppers.js';
export { ScoreThresholdStopper } from './stoppers.js';
export { NoImprovementStopper } from './stoppers.js';
export { SignalStopper } from './stoppers.js';
export { MaxTrackedCandidatesStopper } from './stoppers.js';
export { STR_CANDIDATE_KEY, SINGLE_INSTANCE_SENTINEL } from './types.js';
export {
  SeededRandom,
  idxmax,
  is_dominated,
  json_default,
  remove_dominated_programs,
  find_dominator_programs,
  select_program_candidate_from_pareto_front,
} from './utils.js';
export {
  LM,
  TrackingLM,
  ensure_tracking_lm,
  make_litellm_lm,
} from './lm.js';
export type {
  LM as LMType,
  LMCompletionHook,
  LMCompletionRequest,
  LMCompletionResponse,
  LMOptions,
  TrackingLM as TrackingLMType,
} from './lm.js';
export { Image, _guess_media_type } from './image.js';
export type { ImageOptions, OpenAIImageContentPart } from './image.js';
export {
  ExperimentTracker,
  create_experiment_tracker,
  log_detailed_metrics_after_discovering_new_program,
} from './logging/index.js';
export type {
  ExperimentTrackerOptions,
  LoggedSummary,
  LoggedTable,
} from './logging/index.js';
export {
  CodeExecutionResult,
  ExecutionMode,
  TimeLimitError,
  execute_code,
  get_code_hash,
} from './code_execution.js';
export type { ExecuteCodeOptions, CodeExecutionResultInit } from './code_execution.js';
export {
  StreamCaptureManager,
  ThreadLocalStreamCapture,
  stream_manager,
} from './utils/stdio_capture.js';
export {
  candidate_tree_dot_from_data,
  candidate_tree_html_from_data,
} from './visualization.js';
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
  EvaluationPolicy,
  DataLoader,
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
export { result_from_dict } from './result.js';
export type { GEPAResult } from './result.js';
export {
  GenericRAGAdapter,
  RAGPipeline,
  RAGEvaluationMetrics,
  VectorStoreInterface,
  ChromaVectorStore,
  LanceDBVectorStore,
  MilvusVectorStore,
  QdrantVectorStore,
  WeaviateVectorStore,
  type EmbeddingFunction,
  type GenerationMetrics,
  type GenericRAGConfig,
  type RAGCallableClient,
  type RAGChatMessage,
  type RAGCompletionClient,
  type RAGDataInst,
  type RAGDocument,
  type RAGLLMClient,
  type RAGMetadataFilter,
  type RAGOutput,
  type RAGPipelineConfig,
  type RAGPipelineResult,
  type RAGTrajectory,
  type RetrievalMetrics,
  type LanceQuery,
  type LanceTable,
  type WeaviateCollection,
} from './adapters/generic_rag_adapter/index.js';
