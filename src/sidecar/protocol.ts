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
  objective: string | null;
  background: string | null;
  config: {
    engine?: {
      max_metric_calls?: number;
      seed?: number;
      best_example_evals_k?: number;
    };
    reflection?: {
      reflection_lm_handle?: string;
      reflection_minibatch_size?: number;
      reflection_prompt_template?: string | Record<string, string>;
    };
    tracking?: Record<string, unknown>;
  };
  evaluator_handle: string;
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
  if (eng.seed !== undefined && typeof eng.seed !== 'number') return false;
  if (eng.best_example_evals_k !== undefined && typeof eng.best_example_evals_k !== 'number') return false;
  return true;
}

function is_reflection_shape(v: unknown): boolean {
  if (typeof v !== 'object' || v === null) return false;
  const ref = v as Record<string, unknown>;
  if (ref.reflection_lm_handle !== undefined && typeof ref.reflection_lm_handle !== 'string') return false;
  if (ref.reflection_minibatch_size !== undefined && typeof ref.reflection_minibatch_size !== 'number') return false;
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
  if (cfg.tracking !== undefined && (typeof cfg.tracking !== 'object' || cfg.tracking === null)) return false;
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
  if (typeof p['evaluator_handle'] !== 'string') return false;
  if (p['reflection_lm_handle'] !== null && typeof p['reflection_lm_handle'] !== 'string') return false;
  if (p['objective'] !== null && typeof p['objective'] !== 'string') return false;
  if (p['background'] !== null && typeof p['background'] !== 'string') return false;
  if (p['dataset'] !== null && !Array.isArray(p['dataset'])) return false;
  if (p['valset'] !== null && !Array.isArray(p['valset'])) return false;
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
