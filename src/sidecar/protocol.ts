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
    };
    reflection?: {
      reflection_lm_handle?: string;
      reflection_minibatch_size?: number;
    };
    tracking?: Record<string, unknown>;
  };
  evaluator_handle: string;
  reflection_lm_handle: string | null;
}

export interface CallbackInvokeParams {
  handle: string;
  args: unknown[];
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

export function is_optimize_request(x: unknown): x is JsonRpcRequest<OptimizeRequestParams> {
  if (typeof x !== 'object' || x === null) return false;
  if (!has_string_prop(x, 'jsonrpc') || x.jsonrpc !== '2.0') return false;
  if (!has_id(x)) return false;
  if (!has_string_prop(x, 'method') || x.method !== 'optimize_anything') return false;
  if (!('params' in x)) return false;
  const params = (x as Record<string, unknown>)['params'];
  if (typeof params !== 'object' || params === null) return false;
  if (!has_string_prop(params as object, 'evaluator_handle')) return false;
  return true;
}
