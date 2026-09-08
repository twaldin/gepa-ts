import * as net from 'node:net';
import * as readline from 'node:readline';
import { AsyncLocalStorage } from 'node:async_hooks';
import { optimize_anything } from '../index.js';
import { EvaluatorWrapper } from '../evaluator_wrapper.js';
import { LogContext, runWithLogContext } from '../log_context.js';
import type { Evaluator, LanguageModel, GEPAConfig, EvalResult, SideInfo, DataId, GEPACallback, GEPAAdapter, Candidate, EvaluationBatch, DataLoader, EvaluationPolicy, ProgramIdx } from '../types.js';
import type { JsonRpcResponse, CallbackInvokeParams, EvaluatorCtx } from './protocol.js';
import { is_json_rpc_response, is_optimize_request } from './protocol.js';
import type { GEPAResult } from '../result.js';
import type { GEPAState } from '../state.js';

let _cb_counter = 0;
function next_cb_id(): string { return `srv-${++_cb_counter}`; }

function to_side_info(val: unknown): SideInfo { return val !== null && typeof val === 'object' ? (val as Record<string, unknown>) : {}; }

function parse_eval_result(raw: unknown): EvalResult {
  if (typeof raw === 'number') return raw;
  if (Array.isArray(raw) && typeof raw[0] === 'number') {
    const side = raw.length >= 3 ? raw[2] : raw[1];
    return [raw[0], to_side_info(side)];
  }
  throw new Error(`Unexpected evaluator result: ${JSON.stringify(raw)}`);
}

function is_record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function to_evaluation_batch(raw: unknown): EvaluationBatch {
  if (!is_record(raw) || !Array.isArray(raw['outputs']) || !Array.isArray(raw['scores'])) {
    throw new Error(`Unexpected adapter evaluation result: ${JSON.stringify(raw)}`);
  }
  return {
    outputs: raw['outputs'],
    scores: raw['scores'].map((score) => {
      if (typeof score !== 'number') throw new Error(`Adapter score must be number, got ${typeof score}`);
      return score;
    }),
    ...(Array.isArray(raw['trajectories']) ? { trajectories: raw['trajectories'] } : {}),
    ...(Array.isArray(raw['side_infos']) ? { side_infos: raw['side_infos'].map(to_side_info) } : {}),
    ...(Array.isArray(raw['objective_scores'])
      ? {
          objective_scores: raw['objective_scores'].map((entry) => {
            const scores: Record<string, number> = {};
            if (is_record(entry)) {
              for (const [key, value] of Object.entries(entry)) {
                if (typeof value === 'number') scores[key] = value;
              }
            }
            return scores;
          }),
        }
      : {}),
    ...(typeof raw['num_metric_calls'] === 'number' ? { num_metric_calls: raw['num_metric_calls'] } : {}),
  };
}

function to_reflective_dataset(raw: unknown): Record<string, Array<Record<string, unknown>>> {
  if (!is_record(raw)) return {};
  const out: Record<string, Array<Record<string, unknown>>> = {};
  for (const [component, examples] of Object.entries(raw)) {
    if (!Array.isArray(examples)) continue;
    out[component] = examples.map((example) => is_record(example) ? example : { value: example });
  }
  return out;
}

function to_candidate(raw: unknown): Candidate {
  if (!is_record(raw)) {
    throw new Error(`Adapter propose_new_texts must return an object, got ${typeof raw}`);
  }
  const out: Candidate = {};
  for (const [key, value] of Object.entries(raw)) {
    if (typeof value !== 'string') {
      throw new Error(`Adapter propose_new_texts value for ${key} must be string, got ${typeof value}`);
    }
    out[key] = value;
  }
  return out;
}

function serialize_result(result: GEPAResult): Record<string, unknown> {
  return {
    ...result.to_dict(),
    best_candidate: result.best_candidate,
    num_candidates: result.num_candidates,
    num_val_instances: result.num_val_instances,
  };
}

type RemoteCall = (handle: string, method: string, args: unknown[]) => Promise<unknown>;

function to_data_ids(raw: unknown): DataId[] {
  if (!Array.isArray(raw)) {
    throw new Error(`Loader all_ids must return an array, got ${typeof raw}`);
  }
  return raw.map((item) => {
    if (typeof item !== 'string' && typeof item !== 'number') {
      throw new Error(`Loader id must be string or number, got ${typeof item}`);
    }
    return item;
  });
}

class RemoteCallbackDataLoader implements DataLoader<DataId, unknown> {
  private ids: DataId[] = [];
  private readonly cache = new Map<DataId, unknown>();

  constructor(
    private readonly handle: string,
    private readonly send_remote_call: RemoteCall,
  ) {}

  all_ids(): DataId[] {
    return [...this.ids];
  }

  get length(): number {
    return this.ids.length;
  }

  async refresh(): Promise<void> {
    this.ids = to_data_ids(await this.send_remote_call(this.handle, 'all_ids', []));
  }

  fetch(ids: DataId[]): unknown[] {
    return ids.map((id) => {
      if (!this.cache.has(id)) {
        throw new Error(`Loader id ${String(id)} was not prefetched; use fetch_async for remote loaders.`);
      }
      return this.cache.get(id);
    });
  }

  async fetch_async(ids: DataId[]): Promise<unknown[]> {
    const raw = await this.send_remote_call(this.handle, 'fetch', [ids]);
    if (!Array.isArray(raw)) {
      throw new Error(`Loader fetch must return an array, got ${typeof raw}`);
    }
    if (raw.length !== ids.length) {
      throw new Error(`Loader fetch returned ${raw.length} items for ${ids.length} ids`);
    }
    for (let idx = 0; idx < ids.length; idx += 1) {
      const id = ids[idx];
      if (id !== undefined) {
        this.cache.set(id, raw[idx]);
      }
    }
    await this.refresh();
    return raw;
  }
}

function to_program_idx(raw: unknown): ProgramIdx {
  if (typeof raw !== 'number' || !Number.isInteger(raw)) {
    throw new Error(`Evaluation policy program index must be integer, got ${typeof raw}`);
  }
  return raw;
}

function to_number(raw: unknown): number {
  if (typeof raw !== 'number') {
    throw new Error(`Evaluation policy score must be number, got ${typeof raw}`);
  }
  return raw;
}

function number_map_to_record(map: Map<DataId, number>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [key, value] of map.entries()) {
    out[String(key)] = value;
  }
  return out;
}

function number_array_map_to_record(map: Map<DataId, number[]>): Record<string, number[]> {
  const out: Record<string, number[]> = {};
  for (const [key, value] of map.entries()) {
    out[String(key)] = [...value];
  }
  return out;
}

function policy_state_snapshot(state: GEPAState): Record<string, unknown> {
  return {
    program_candidates: state.program_candidates.map((candidate) => ({ ...candidate })),
    prog_candidate_val_subscores: state.prog_candidate_val_subscores.map(number_map_to_record),
    prog_candidate_objective_scores: state.prog_candidate_objective_scores.map((scores) => ({ ...scores })),
    parent_program_for_candidate: state.parent_program_for_candidate.map((parents) => [...parents]),
    valset_evaluations: number_array_map_to_record(state.valset_evaluations),
    i: state.i,
    num_full_ds_evals: state.num_full_ds_evals,
    total_num_evals: state.total_num_evals,
    num_metric_calls_by_discovery: [...state.num_metric_calls_by_discovery],
  };
}

class RemoteEvaluationPolicy implements EvaluationPolicy<DataId, unknown> {
  constructor(
    private readonly handle: string,
    private readonly send_remote_call: RemoteCall,
  ) {}

  async get_eval_batch(loader: DataLoader<DataId, unknown>, state: GEPAState, target_program_idx?: ProgramIdx): Promise<DataId[]> {
    return to_data_ids(
      await this.send_remote_call(this.handle, 'get_eval_batch', [
        loader.all_ids(),
        policy_state_snapshot(state),
        target_program_idx ?? null,
      ]),
    );
  }

  async get_best_program(state: GEPAState): Promise<ProgramIdx> {
    return to_program_idx(await this.send_remote_call(this.handle, 'get_best_program', [policy_state_snapshot(state)]));
  }

  async get_valset_score(program_idx: ProgramIdx, state: GEPAState): Promise<number> {
    return to_number(
      await this.send_remote_call(this.handle, 'get_valset_score', [
        program_idx,
        policy_state_snapshot(state),
      ]),
    );
  }
}

function build_remote_callback(handle: string, methods: string[], send: (h: string, m: string, a: unknown[]) => void): GEPACallback {
  const cb: Partial<Required<GEPACallback>> = {};
  for (const method of methods) {
    switch (method) {
      case 'on_optimization_start': cb.on_optimization_start = (e) => { send(handle, method, [e]); }; break;
      case 'on_optimization_end': cb.on_optimization_end = (e) => { send(handle, method, [e]); }; break;
      case 'on_iteration_start': cb.on_iteration_start = (e) => { send(handle, method, [e]); }; break;
      case 'on_iteration_end': cb.on_iteration_end = (e) => { send(handle, method, [e]); }; break;
      case 'on_candidate_selected': cb.on_candidate_selected = (e) => { send(handle, method, [e]); }; break;
      case 'on_minibatch_sampled': cb.on_minibatch_sampled = (e) => { send(handle, method, [e]); }; break;
      case 'on_evaluation_start': cb.on_evaluation_start = (e) => { send(handle, method, [e]); }; break;
      case 'on_evaluation_end': cb.on_evaluation_end = (e) => { send(handle, method, [e]); }; break;
      case 'on_evaluation_skipped': cb.on_evaluation_skipped = (e) => { send(handle, method, [e]); }; break;
      case 'on_valset_evaluated': cb.on_valset_evaluated = (e) => { send(handle, method, [e]); }; break;
      case 'on_reflective_dataset_built': cb.on_reflective_dataset_built = (e) => { send(handle, method, [e]); }; break;
      case 'on_proposal_start': cb.on_proposal_start = (e) => { send(handle, method, [e]); }; break;
      case 'on_proposal_end': cb.on_proposal_end = (e) => { send(handle, method, [e]); }; break;
      case 'on_candidate_accepted': cb.on_candidate_accepted = (e) => { send(handle, method, [e]); }; break;
      case 'on_candidate_rejected': cb.on_candidate_rejected = (e) => { send(handle, method, [e]); }; break;
      case 'on_merge_attempted': cb.on_merge_attempted = (e) => { send(handle, method, [e]); }; break;
      case 'on_merge_accepted': cb.on_merge_accepted = (e) => { send(handle, method, [e]); }; break;
      case 'on_merge_rejected': cb.on_merge_rejected = (e) => { send(handle, method, [e]); }; break;
      case 'on_pareto_front_updated': cb.on_pareto_front_updated = (e) => { send(handle, method, [e]); }; break;
      case 'on_state_saved': cb.on_state_saved = (e) => { send(handle, method, [e]); }; break;
      case 'on_budget_updated': cb.on_budget_updated = (e) => { send(handle, method, [e]); }; break;
      case 'on_error': cb.on_error = (e) => { send(handle, method, [e]); }; break;
    }
  }
  return cb;
}

interface WrapperConfig {
  evaluatorHandle: string;
  singleInstanceMode: boolean;
  captureStdio: boolean;
  strCandidateMode: boolean;
  raiseOnException: boolean;
}
const _logContexts = new Map<string, LogContext>();
const _threadContexts = new Map<number, LogContext | null>();
const _wrappers = new Map<string, WrapperConfig>();
let _handleCounter = 0;
const _callLogCtxHandleAls = new AsyncLocalStorage<string | null>();
function _nextHandle(prefix: string): string { _handleCounter += 1; return `${prefix}-${_handleCounter}`; }

export function create_server(): net.Server {
  return net.createServer((socket: net.Socket) => {
    const pending = new Map<string | number, (r: JsonRpcResponse) => void>();
    const logContexts = _logContexts;
    const threadContexts = _threadContexts;
    const wrappers = _wrappers;
    const rl = readline.createInterface({ input: socket, crlfDelay: Infinity });
    const callLogCtxHandleAls = _callLogCtxHandleAls;

    function nextHandle(prefix: string): string { return _nextHandle(prefix); }
    function write_line(obj: unknown): void { socket.write(JSON.stringify(obj) + '\n'); }

    function send_remote_call(handle: string, method: string, args: unknown[]): Promise<unknown> {
      const id = next_cb_id();
      const log_ctx_handle = callLogCtxHandleAls.getStore() ?? null;
      return new Promise<unknown>((resolve, reject) => {
        pending.set(id, (r: JsonRpcResponse) => r.error !== undefined ? reject(new Error(r.error.message)) : resolve(r.result));
        const cbParams: CallbackInvokeParams & { log_ctx_handle?: string } = { handle, method, args };
        if (log_ctx_handle !== null) cbParams.log_ctx_handle = log_ctx_handle;
        write_line({ jsonrpc: '2.0', id, method: 'callback_invoke', params: cbParams });
      });
    }

    function getThreadCtx(params: Record<string, unknown>): LogContext | null {
      const tid = params['client_thread_id'];
      if (typeof tid !== 'number') return null;
      return threadContexts.get(tid) ?? null;
    }

    rl.on('line', (line: string) => {
      let parsed: unknown;
      try { parsed = JSON.parse(line); } catch { return; }

      if (is_json_rpc_response(parsed)) {
        const resolver = pending.get(parsed.id);
        if (resolver !== undefined) { pending.delete(parsed.id); resolver(parsed); }
        return;
      }

      const req = parsed as Record<string, unknown>;
      const id = req['id'] ?? 0;
      const method = req['method'];
      const params = (req['params'] ?? {}) as Record<string, unknown>;
      if (req['jsonrpc'] !== '2.0' || (typeof method !== 'string')) {
        write_line({ jsonrpc: '2.0', id, error: { code: -32600, message: 'Invalid Request' } });
        return;
      }

      runWithLogContext(getThreadCtx(params), () => {
        if (method === 'optimize_anything') {
          if (!is_optimize_request(parsed)) {
            write_line({ jsonrpc: '2.0', id, error: { code: -32600, message: 'Invalid Request' } });
            return;
          }
          const p = parsed.params;
          const evaluator = p.evaluator_handle !== undefined
            ? (((candidate: string | Record<string, string>, ctx?: EvaluatorCtx): Promise<EvalResult> => {
                const args: unknown[] = ctx !== undefined ? [candidate, { example: ctx.example, opt_state: ctx.opt_state }] : [candidate];
                return send_remote_call(p.evaluator_handle!, '__call__', args).then(parse_eval_result);
              }) as Evaluator)
            : undefined;

          const adapter: GEPAAdapter | undefined = typeof p.adapter_handle === 'string'
            ? {
                evaluate: (batch: unknown[], candidate: Candidate, capture_traces: boolean = false) =>
                  send_remote_call(p.adapter_handle!, 'evaluate', [batch, candidate, capture_traces]).then(to_evaluation_batch),
                make_reflective_dataset: (candidate: Candidate, eval_batch: EvaluationBatch, components_to_update: string[]) =>
                  send_remote_call(
                    p.adapter_handle!,
                    'make_reflective_dataset',
                    [candidate, eval_batch, components_to_update],
                  ).then(to_reflective_dataset),
                ...(p.adapter_propose_new_texts === true
                  ? {
                      propose_new_texts: (candidate: Candidate, reflective_dataset: Record<string, Array<Record<string, unknown>>>, components_to_update: string[]) =>
                        send_remote_call(
                          p.adapter_handle!,
                          'propose_new_texts',
                          [candidate, reflective_dataset, components_to_update],
                        ).then(to_candidate),
                    }
                  : {}),
              }
            : undefined;

          const reflection_lm: LanguageModel | undefined = p.reflection_lm_handle !== null
            ? (prompt) => send_remote_call(p.reflection_lm_handle as string, '__call__', [prompt]).then((r) => {
                if (typeof r !== 'string') throw new Error(`LM must return string, got ${typeof r}`);
                return r;
              })
            : undefined;

          const raw_engine = p.config.engine ?? {};
          const raw_reflection = p.config.reflection ?? {};
          const raw_acceptance_criterion = raw_engine.acceptance_criterion;
          const acceptance_criterion =
            raw_acceptance_criterion === 'strict_improvement' || raw_acceptance_criterion === 'improvement_or_equal'
              ? raw_acceptance_criterion
              : undefined;
          const raw_candidate_selection_strategy = raw_engine.candidate_selection_strategy;
          const candidate_selection_strategy =
            raw_candidate_selection_strategy === 'pareto' ||
            raw_candidate_selection_strategy === 'current_best' ||
            raw_candidate_selection_strategy === 'epsilon_greedy' ||
            raw_candidate_selection_strategy === 'top_k_pareto'
              ? raw_candidate_selection_strategy
              : undefined;
          const raw_frontier_type = raw_engine.frontier_type;
          const frontier_type =
            raw_frontier_type === 'instance' ||
            raw_frontier_type === 'objective' ||
            raw_frontier_type === 'hybrid' ||
            raw_frontier_type === 'cartesian'
              ? raw_frontier_type
              : undefined;
          const val_evaluation_policy = typeof p.val_evaluation_policy_handle === 'string'
            ? new RemoteEvaluationPolicy(p.val_evaluation_policy_handle, send_remote_call)
            : undefined;
          const engine_config: NonNullable<GEPAConfig['engine']> = {
            ...(typeof raw_engine.max_metric_calls === 'number' ? { max_metric_calls: raw_engine.max_metric_calls } : {}),
            ...(typeof raw_engine.max_candidate_proposals === 'number' ? { max_candidate_proposals: raw_engine.max_candidate_proposals } : {}),
            ...(typeof raw_engine.max_reflection_cost === 'number' ? { max_reflection_cost: raw_engine.max_reflection_cost } : {}),
            ...(typeof raw_engine.seed === 'number' ? { seed: raw_engine.seed } : {}),
            ...(typeof raw_engine.raise_on_exception === 'boolean' ? { raise_on_exception: raw_engine.raise_on_exception } : {}),
            ...(typeof raw_engine.track_best_outputs === 'boolean' ? { track_best_outputs: raw_engine.track_best_outputs } : {}),
            ...(typeof raw_engine.best_example_evals_k === 'number' ? { best_example_evals_k: raw_engine.best_example_evals_k } : {}),
            ...(typeof raw_engine.run_dir === 'string' ? { run_dir: raw_engine.run_dir } : {}),
            ...(typeof raw_engine.cache_evaluation === 'boolean' ? { cache_evaluation: raw_engine.cache_evaluation } : {}),
            ...(typeof raw_engine.cache_evaluation_storage === 'string' ? { cache_evaluation_storage: raw_engine.cache_evaluation_storage } : {}),
            ...(typeof raw_engine.capture_stdio === 'boolean' ? { capture_stdio: raw_engine.capture_stdio } : {}),
            ...(acceptance_criterion !== undefined ? { acceptance_criterion } : {}),
            ...(candidate_selection_strategy !== undefined ? { candidate_selection_strategy } : {}),
            ...(frontier_type !== undefined ? { frontier_type } : {}),
            ...(val_evaluation_policy !== undefined ? { val_evaluation_policy } : {}),
          };
          const reflection_config: GEPAConfig['reflection'] = {
            ...(reflection_lm !== undefined ? { reflection_lm } : {}),
            ...(typeof raw_reflection.reflection_minibatch_size === 'number' ? { reflection_minibatch_size: raw_reflection.reflection_minibatch_size } : {}),
            ...(raw_reflection.reflection_prompt_template !== undefined ? { reflection_prompt_template: raw_reflection.reflection_prompt_template as string | Record<string, string> } : {}),
            ...(raw_reflection.module_selector === 'round_robin' || raw_reflection.module_selector === 'all'
              ? { module_selector: raw_reflection.module_selector }
              : {}),
            ...(raw_reflection.batch_sampler === 'epoch_shuffled' ? { batch_sampler: raw_reflection.batch_sampler } : {}),
            ...(typeof raw_reflection.skip_perfect_score === 'boolean'
              ? { skip_perfect_score: raw_reflection.skip_perfect_score }
              : {}),
            ...(typeof raw_reflection.perfect_score === 'number' ? { perfect_score: raw_reflection.perfect_score } : {}),
          };

          const callback_handles = p.callback_handles ?? [];
          const callbacks: GEPACallback[] = callback_handles.map(({ id: h, methods }) =>
            build_remote_callback(h, methods, (handle, m, args) => { send_remote_call(handle, m, args).catch(() => undefined); }),
          );
          const raw_refiner = p.config.refiner;
          const refiner_lm: LanguageModel | undefined =
            typeof raw_refiner === 'object' &&
            raw_refiner !== null &&
            typeof raw_refiner.refiner_lm_handle === 'string'
              ? (prompt) => send_remote_call(raw_refiner.refiner_lm_handle as string, '__call__', [prompt]).then((r) => {
                  if (typeof r !== 'string') throw new Error(`Refiner LM must return string, got ${typeof r}`);
                  return r;
                })
              : undefined;
          const refiner_config: GEPAConfig['refiner'] | undefined =
            raw_refiner === null
              ? null
              : typeof raw_refiner === 'object' && raw_refiner !== null
                ? {
                    ...('max_refinements' in raw_refiner && typeof raw_refiner.max_refinements === 'number'
                      ? { max_refinements: raw_refiner.max_refinements }
                      : {}),
                    ...(refiner_lm !== undefined ? { refiner_lm } : {}),
                  }
                : undefined;
          const raw_merge = p.config.merge;
          const merge_config: GEPAConfig['merge'] | undefined =
            raw_merge === null
              ? null
              : typeof raw_merge === 'object' && raw_merge !== null
                ? {
                    ...('max_merge_invocations' in raw_merge && typeof raw_merge.max_merge_invocations === 'number'
                      ? { max_merge_invocations: raw_merge.max_merge_invocations }
                      : {}),
                    ...('merge_val_overlap_floor' in raw_merge && typeof raw_merge.merge_val_overlap_floor === 'number'
                      ? { merge_val_overlap_floor: raw_merge.merge_val_overlap_floor }
                      : {}),
                  }
                : undefined;
          const raw_tracking = p.config.tracking;
          const tracking_config: GEPAConfig['tracking'] | undefined =
            typeof raw_tracking === 'object' && raw_tracking !== null
              ? {
                  ...('use_wandb' in raw_tracking && typeof raw_tracking.use_wandb === 'boolean'
                    ? { use_wandb: raw_tracking.use_wandb }
                    : {}),
                  ...('wandb_api_key' in raw_tracking && typeof raw_tracking.wandb_api_key === 'string'
                    ? { wandb_api_key: raw_tracking.wandb_api_key }
                    : {}),
                  ...('wandb_init_kwargs' in raw_tracking && typeof raw_tracking.wandb_init_kwargs === 'object' && raw_tracking.wandb_init_kwargs !== null && !Array.isArray(raw_tracking.wandb_init_kwargs)
                    ? { wandb_init_kwargs: raw_tracking.wandb_init_kwargs as Record<string, unknown> }
                    : {}),
                  ...('wandb_attach_existing' in raw_tracking && typeof raw_tracking.wandb_attach_existing === 'boolean'
                    ? { wandb_attach_existing: raw_tracking.wandb_attach_existing }
                    : {}),
                  ...('wandb_step_metric' in raw_tracking && typeof raw_tracking.wandb_step_metric === 'string'
                    ? { wandb_step_metric: raw_tracking.wandb_step_metric }
                    : {}),
                  ...('use_mlflow' in raw_tracking && typeof raw_tracking.use_mlflow === 'boolean'
                    ? { use_mlflow: raw_tracking.use_mlflow }
                    : {}),
                  ...('mlflow_tracking_uri' in raw_tracking && typeof raw_tracking.mlflow_tracking_uri === 'string'
                    ? { mlflow_tracking_uri: raw_tracking.mlflow_tracking_uri }
                    : {}),
                  ...('mlflow_experiment_name' in raw_tracking && typeof raw_tracking.mlflow_experiment_name === 'string'
                    ? { mlflow_experiment_name: raw_tracking.mlflow_experiment_name }
                    : {}),
                  ...('mlflow_attach_existing' in raw_tracking && typeof raw_tracking.mlflow_attach_existing === 'boolean'
                    ? { mlflow_attach_existing: raw_tracking.mlflow_attach_existing }
                    : {}),
                  ...('key_prefix' in raw_tracking && typeof raw_tracking.key_prefix === 'string'
                    ? { key_prefix: raw_tracking.key_prefix }
                    : {}),
                }
              : undefined;
          const gepa_config: GEPAConfig = {
            engine: engine_config,
            reflection: reflection_config,
            ...(tracking_config !== undefined ? { tracking: tracking_config } : {}),
            ...(merge_config !== undefined ? { merge: merge_config } : {}),
            ...(refiner_config !== undefined ? { refiner: refiner_config } : {}),
            ...(callbacks.length > 0 ? { callbacks } : {}),
          };

          const dataset = typeof p.dataset_loader_handle === 'string'
            ? new RemoteCallbackDataLoader(p.dataset_loader_handle, send_remote_call)
            : p.dataset;
          const valset = typeof p.valset_loader_handle === 'string'
            ? new RemoteCallbackDataLoader(p.valset_loader_handle, send_remote_call)
            : p.valset;

          optimize_anything({
            seed_candidate: p.seed_candidate,
            ...(evaluator !== undefined ? { evaluator } : {}),
            ...(adapter !== undefined ? { adapter } : {}),
            dataset,
            valset,
            objective: p.objective,
            background: p.background,
            config: gepa_config,
          }).then((result) => {
            write_line({ jsonrpc: '2.0', id, result: serialize_result(result) });
          }).catch((err: unknown) => {
            const message = err instanceof Error ? err.message : String(err);
            write_line({ jsonrpc: '2.0', id, error: { code: -32000, message } });
          });
          return;
        }

        if (method === 'log_context.create') {
          const handle = nextHandle('logctx');
          logContexts.set(handle, new LogContext());
          write_line({ jsonrpc: '2.0', id, result: handle });
          return;
        }
        if (method === 'log_context.write') {
          const handle = params['handle'];
          const text = params['text'];
          if (typeof handle !== 'string' || typeof text !== 'string') {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params' } });
            return;
          }
          const ctx = logContexts.get(handle);
          if (ctx === undefined) {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params' } });
            return;
          }
          ctx.write(text);
          write_line({ jsonrpc: '2.0', id, result: null });
          return;
        }
        if (method === 'log_context.drain') {
          const handle = params['handle'];
          if (typeof handle !== 'string') {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params' } });
            return;
          }
          const ctx = logContexts.get(handle);
          if (ctx === undefined) {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params' } });
            return;
          }
          write_line({ jsonrpc: '2.0', id, result: ctx.drain() });
          return;
        }
        if (method === 'set_log_context') {
          const tid = params['client_thread_id'];
          const handle = params['handle'];
          if (typeof tid !== 'number') {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'client_thread_id required' } });
            return;
          }
          threadContexts.set(tid, handle === null ? null : (typeof handle === 'string' ? (logContexts.get(handle) ?? null) : null));
          write_line({ jsonrpc: '2.0', id, result: null });
          return;
        }
        if (method === 'evaluator_wrapper.create') {
          const evaluatorHandle = params['evaluator_handle'];
          const singleInstanceMode = params['single_instance_mode'];
          if (typeof evaluatorHandle !== 'string' || typeof singleInstanceMode !== 'boolean') {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params' } });
            return;
          }
          const handle = nextHandle('evalwrap');
          wrappers.set(handle, {
            evaluatorHandle,
            singleInstanceMode,
            captureStdio: params['capture_stdio'] === true,
            strCandidateMode: params['str_candidate_mode'] === true,
            raiseOnException: params['raise_on_exception'] !== false,
          });
          write_line({ jsonrpc: '2.0', id, result: handle });
          return;
        }
        if (method === 'evaluator_wrapper.call') {
          const handle = params['handle'];
          const candidate = params['candidate'];
          if (typeof handle !== 'string' || typeof candidate !== 'object' || candidate === null) {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params' } });
            return;
          }
          const cfg = wrappers.get(handle);
          if (cfg === undefined) {
            write_line({ jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params' } });
            return;
          }
          const opt = params['opt_state'];
          const best =
            typeof opt === 'object' && opt !== null && Array.isArray((opt as { best_example_evals?: unknown }).best_example_evals)
              ? ((opt as { best_example_evals: Array<{ score: number; side_info: Record<string, unknown> }> }).best_example_evals)
              : [];
          const evaluator: Evaluator = ((c: string | Record<string, string>, ctx?: EvaluatorCtx) => {
            const args: unknown[] = ctx !== undefined ? [c, ctx] : [c];
            return send_remote_call(cfg.evaluatorHandle, '__call__', args).then(parse_eval_result);
          }) as Evaluator;
          const wrapper = new EvaluatorWrapper(
            evaluator,
            cfg.singleInstanceMode,
            cfg.captureStdio,
            cfg.strCandidateMode,
            cfg.raiseOnException,
          );
          const callLogContext = new LogContext();
          const callLogCtxHandle = nextHandle('logctx');
          logContexts.set(callLogCtxHandle, callLogContext);
          callLogCtxHandleAls.run(callLogCtxHandle, () => {
            wrapper.call(candidate as Record<string, string>, params['example'], { best_example_evals: best }, callLogContext).then((ret) => {
              write_line({ jsonrpc: '2.0', id, result: ret });
            }).catch((err: unknown) => {
              const message = err instanceof Error ? err.message : String(err);
              write_line({ jsonrpc: '2.0', id, error: { code: -32000, message } });
            }).finally(() => {
              logContexts.delete(callLogCtxHandle);
            });
          });
          return;
        }

        write_line({ jsonrpc: '2.0', id, error: { code: -32601, message: 'Method not found' } });
      });
    });

    socket.on('error', () => rl.close());
  });
}
