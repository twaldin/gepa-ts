export interface JsonRpcRequest<P = unknown> {
  jsonrpc: '2.0';
  id: string | number;
  method: string;
  params: P;
}

export interface JsonRpcResponse<R = unknown> {
  jsonrpc: '2.0';
  id: string | number;
  result?: R;
  error?: JsonRpcError;
}

export interface JsonRpcError {
  code: number;
  message: string;
  data?: unknown;
}

export interface CallbackHandle {
  id: string;
  methods: string[];
}

export interface OptimizeRequestParams {
  seed_candidate: string | Record<string, string> | null;
  dataset: unknown[] | null;
  valset: unknown[] | null;
  dataset_loader_handle?: string | null;
  valset_loader_handle?: string | null;
  objective: string | null;
  background: string | null;
  config: {
    engine?: {
      max_metric_calls?: number;
      max_candidate_proposals?: number;
      max_reflection_cost?: number;
      seed?: number;
      raise_on_exception?: boolean;
      track_best_outputs?: boolean;
      best_example_evals_k?: number;
      run_dir?: string;
      cache_evaluation?: boolean;
      cache_evaluation_storage?: string;
      capture_stdio?: boolean;
      frontier_type?: string;
      acceptance_criterion?: string;
      candidate_selection_strategy?: string;
    };
    reflection?: {
      reflection_lm_handle?: string;
      reflection_minibatch_size?: number;
      reflection_prompt_template?: string | Record<string, string>;
      module_selector?: string;
      batch_sampler?: string;
      skip_perfect_score?: boolean;
      perfect_score?: number;
    };
    tracking?: {
      use_wandb?: boolean;
      wandb_api_key?: string | null;
      wandb_init_kwargs?: Record<string, unknown> | null;
      wandb_attach_existing?: boolean;
      wandb_step_metric?: string | null;
      use_mlflow?: boolean;
      mlflow_tracking_uri?: string | null;
      mlflow_experiment_name?: string | null;
      mlflow_attach_existing?: boolean;
      key_prefix?: string | null;
    };
    merge?: {
      max_merge_invocations?: number;
      merge_val_overlap_floor?: number;
    } | null;
    refiner?: {
      refiner_lm_handle?: string;
      max_refinements?: number;
    } | null;
  };
  evaluator_handle?: string;
  adapter_handle?: string | null;
  adapter_propose_new_texts?: boolean;
  val_evaluation_policy_handle?: string | null;
  reflection_lm_handle: string | null;
  callback_handles?: CallbackHandle[];
}

export interface CallbackInvokeParams {
  handle: string;
  method: string;
  args: unknown[];
}

export interface ThreadedParams {
  client_thread_id?: number;
}

export interface LogContextWriteParams extends ThreadedParams {
  handle: string;
  text: string;
}

export interface LogContextDrainParams extends ThreadedParams {
  handle: string;
}

export interface SetLogContextParams extends ThreadedParams {
  handle: string | null;
}

export interface EvaluatorWrapperCreateParams extends ThreadedParams {
  evaluator_handle: string;
  single_instance_mode: boolean;
  capture_stdio?: boolean;
  str_candidate_mode?: boolean;
  raise_on_exception?: boolean;
}

export interface EvaluatorWrapperCallParams extends ThreadedParams {
  handle: string;
  candidate: string | Record<string, string>;
  example?: unknown;
  opt_state?: EvaluatorCtx['opt_state'];
}

export interface EvaluatorCtx {
  example?: unknown;
  opt_state?: {
    best_example_evals: Array<{ score: number; side_info: Record<string, unknown> }>;
  };
}

function has_string_prop<K extends string>(
  obj: object,
  key: K,
): obj is Record<K, string> {
  return key in obj && typeof (obj as Record<string, unknown>)[key] === 'string';
}

function has_id(obj: object): obj is { id: string | number } {
  if (!('id' in obj)) return false;
  const id = (obj as Record<string, unknown>)['id'];
  return typeof id === 'string' || typeof id === 'number';
}

export function is_json_rpc_response(x: unknown): x is JsonRpcResponse {
  if (typeof x !== 'object' || x === null) return false;
  if (!has_string_prop(x, 'jsonrpc') || x.jsonrpc !== '2.0') return false;
  if (!has_id(x)) return false;
  return 'result' in x || 'error' in x;
}

function is_seed_candidate(v: unknown): boolean {
  if (v === null || typeof v === 'string') return true;
  if (typeof v !== 'object') return false;
  for (const k in v as Record<string, unknown>) {
    if (typeof (v as Record<string, unknown>)[k] !== 'string') return false;
  }
  return true;
}

function is_engine_shape(v: unknown): boolean {
  if (typeof v !== 'object' || v === null) return false;
  const eng = v as Record<string, unknown>;
  if (eng.max_metric_calls !== undefined && typeof eng.max_metric_calls !== 'number') return false;
  if (eng.max_candidate_proposals !== undefined && typeof eng.max_candidate_proposals !== 'number') return false;
  if (eng.max_reflection_cost !== undefined && typeof eng.max_reflection_cost !== 'number') return false;
  if (eng.seed !== undefined && typeof eng.seed !== 'number') return false;
  if (eng.raise_on_exception !== undefined && typeof eng.raise_on_exception !== 'boolean') return false;
  if (eng.track_best_outputs !== undefined && typeof eng.track_best_outputs !== 'boolean') return false;
  if (eng.best_example_evals_k !== undefined && typeof eng.best_example_evals_k !== 'number') return false;
  if (eng.run_dir !== undefined && typeof eng.run_dir !== 'string') return false;
  if (eng.cache_evaluation !== undefined && typeof eng.cache_evaluation !== 'boolean') return false;
  if (eng.cache_evaluation_storage !== undefined && typeof eng.cache_evaluation_storage !== 'string') return false;
  if (eng.capture_stdio !== undefined && typeof eng.capture_stdio !== 'boolean') return false;
  if (
    eng.frontier_type !== undefined &&
    eng.frontier_type !== 'instance' &&
    eng.frontier_type !== 'objective' &&
    eng.frontier_type !== 'hybrid' &&
    eng.frontier_type !== 'cartesian'
  ) return false;
  if (
    eng.acceptance_criterion !== undefined &&
    eng.acceptance_criterion !== 'strict_improvement' &&
    eng.acceptance_criterion !== 'improvement_or_equal'
  ) return false;
  if (
    eng.candidate_selection_strategy !== undefined &&
    eng.candidate_selection_strategy !== 'pareto' &&
    eng.candidate_selection_strategy !== 'current_best' &&
    eng.candidate_selection_strategy !== 'epsilon_greedy' &&
    eng.candidate_selection_strategy !== 'top_k_pareto'
  ) return false;
  return true;
}

function is_refiner_shape(v: unknown): boolean {
  if (v === null) return true;
  if (typeof v !== 'object') return false;
  const ref = v as Record<string, unknown>;
  if (ref.refiner_lm_handle !== undefined && typeof ref.refiner_lm_handle !== 'string') return false;
  if (ref.max_refinements !== undefined && typeof ref.max_refinements !== 'number') return false;
  return true;
}

function is_merge_shape(v: unknown): boolean {
  if (v === null) return true;
  if (typeof v !== 'object') return false;
  const merge = v as Record<string, unknown>;
  if (merge.max_merge_invocations !== undefined && typeof merge.max_merge_invocations !== 'number') return false;
  if (merge.merge_val_overlap_floor !== undefined && typeof merge.merge_val_overlap_floor !== 'number') return false;
  return true;
}

function is_tracking_shape(v: unknown): boolean {
  if (typeof v !== 'object' || v === null) return false;
  const tracking = v as Record<string, unknown>;
  if (tracking.use_wandb !== undefined && typeof tracking.use_wandb !== 'boolean') return false;
  if (tracking.wandb_api_key !== undefined && tracking.wandb_api_key !== null && typeof tracking.wandb_api_key !== 'string') return false;
  if (
    tracking.wandb_init_kwargs !== undefined &&
    tracking.wandb_init_kwargs !== null &&
    (typeof tracking.wandb_init_kwargs !== 'object' || Array.isArray(tracking.wandb_init_kwargs))
  ) return false;
  if (tracking.wandb_attach_existing !== undefined && typeof tracking.wandb_attach_existing !== 'boolean') return false;
  if (tracking.wandb_step_metric !== undefined && tracking.wandb_step_metric !== null && typeof tracking.wandb_step_metric !== 'string') return false;
  if (tracking.use_mlflow !== undefined && typeof tracking.use_mlflow !== 'boolean') return false;
  if (tracking.mlflow_tracking_uri !== undefined && tracking.mlflow_tracking_uri !== null && typeof tracking.mlflow_tracking_uri !== 'string') return false;
  if (tracking.mlflow_experiment_name !== undefined && tracking.mlflow_experiment_name !== null && typeof tracking.mlflow_experiment_name !== 'string') return false;
  if (tracking.mlflow_attach_existing !== undefined && typeof tracking.mlflow_attach_existing !== 'boolean') return false;
  if (tracking.key_prefix !== undefined && tracking.key_prefix !== null && typeof tracking.key_prefix !== 'string') return false;
  return true;
}

function is_reflection_shape(v: unknown): boolean {
  if (typeof v !== 'object' || v === null) return false;
  const ref = v as Record<string, unknown>;
  if (ref.reflection_lm_handle !== undefined && typeof ref.reflection_lm_handle !== 'string') return false;
  if (ref.reflection_minibatch_size !== undefined && typeof ref.reflection_minibatch_size !== 'number') return false;
  if (
    ref.module_selector !== undefined &&
    ref.module_selector !== 'round_robin' &&
    ref.module_selector !== 'all'
  ) return false;
  if (ref.batch_sampler !== undefined && ref.batch_sampler !== 'epoch_shuffled') return false;
  if (ref.skip_perfect_score !== undefined && typeof ref.skip_perfect_score !== 'boolean') return false;
  if (ref.perfect_score !== undefined && typeof ref.perfect_score !== 'number') return false;
  if (ref.reflection_prompt_template !== undefined) {
    const rpt = ref.reflection_prompt_template;
    if (typeof rpt !== 'string' && (typeof rpt !== 'object' || rpt === null)) return false;
  }
  return true;
}

function is_config_shape(v: unknown): boolean {
  if (typeof v !== 'object' || v === null) return false;
  const cfg = v as Record<string, unknown>;
  if (cfg.engine !== undefined && !is_engine_shape(cfg.engine)) return false;
  if (cfg.reflection !== undefined && !is_reflection_shape(cfg.reflection)) return false;
  if (cfg.tracking !== undefined && !is_tracking_shape(cfg.tracking)) return false;
  if (cfg.merge !== undefined && !is_merge_shape(cfg.merge)) return false;
  if (cfg.refiner !== undefined && !is_refiner_shape(cfg.refiner)) return false;
  return true;
}

function is_callback_handle(v: unknown): v is CallbackHandle {
  if (typeof v !== 'object' || v === null) return false;
  const h = v as Record<string, unknown>;
  if (typeof h['id'] !== 'string') return false;
  if (!Array.isArray(h['methods'])) return false;
  for (const m of h['methods'] as unknown[]) {
    if (typeof m !== 'string') return false;
  }
  return true;
}

export function is_optimize_request(x: unknown): x is JsonRpcRequest<OptimizeRequestParams> {
  if (typeof x !== 'object' || x === null) return false;
  if (!has_string_prop(x, 'jsonrpc') || x.jsonrpc !== '2.0') return false;
  if (!has_id(x)) return false;
  if (!has_string_prop(x, 'method') || x.method !== 'optimize_anything') return false;
  if (!('params' in x)) return false;
  const params = (x as Record<string, unknown>)['params'];
  if (typeof params !== 'object' || params === null) return false;
  const p = params as Record<string, unknown>;
  if (p['evaluator_handle'] !== undefined && typeof p['evaluator_handle'] !== 'string') return false;
  if (p['adapter_handle'] !== undefined && p['adapter_handle'] !== null && typeof p['adapter_handle'] !== 'string') return false;
  if (p['adapter_propose_new_texts'] !== undefined && typeof p['adapter_propose_new_texts'] !== 'boolean') return false;
  if (
    p['val_evaluation_policy_handle'] !== undefined &&
    p['val_evaluation_policy_handle'] !== null &&
    typeof p['val_evaluation_policy_handle'] !== 'string'
  ) return false;
  if (typeof p['evaluator_handle'] !== 'string' && typeof p['adapter_handle'] !== 'string') return false;
  if (p['reflection_lm_handle'] !== null && typeof p['reflection_lm_handle'] !== 'string') return false;
  if (p['objective'] !== null && typeof p['objective'] !== 'string') return false;
  if (p['background'] !== null && typeof p['background'] !== 'string') return false;
  if (p['dataset'] !== null && !Array.isArray(p['dataset'])) return false;
  if (p['valset'] !== null && !Array.isArray(p['valset'])) return false;
  if (p['dataset_loader_handle'] !== undefined && p['dataset_loader_handle'] !== null && typeof p['dataset_loader_handle'] !== 'string') return false;
  if (p['valset_loader_handle'] !== undefined && p['valset_loader_handle'] !== null && typeof p['valset_loader_handle'] !== 'string') return false;
  if (!is_seed_candidate(p['seed_candidate'])) return false;
  if (!is_config_shape(p['config'])) return false;
  if (p['callback_handles'] !== undefined) {
    if (!Array.isArray(p['callback_handles'])) return false;
    for (const h of p['callback_handles'] as unknown[]) {
      if (!is_callback_handle(h)) return false;
    }
  }
  return true;
}
