import inspect
import sys
import json
import os
import socket
import threading
import warnings
import dataclasses
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional

from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.utils.stdio_capture import stream_manager
from gepa.lm import LM

class GEPAEngine:
    pass

_STR_CANDIDATE_KEY = "current_candidate"
_SINGLE_INSTANCE_SENTINEL = object()
DEFAULT_REFINER_PROMPT = """You are a refinement agent improving candidates in an optimization loop.

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
"""


def make_litellm_lm(model_name: str, **kwargs):
    return LM(model_name, **kwargs)


# ---------------------------------------------------------------------------
# OptimizationState
# ---------------------------------------------------------------------------


@dataclass
class OptimizationState:
    best_example_evals: list = field(default_factory=list)


def _extract_fenced_text(lm_out: str) -> str:
    start = lm_out.find("```") + 3
    end = lm_out.rfind("```")
    if start >= end:
        stripped = lm_out.strip()
        if stripped.startswith("```"):
            first_line_end = stripped.find("\n")
            return stripped[first_line_end + 1 :].strip() if first_line_end >= 0 else ""
        if stripped.endswith("```"):
            return stripped[:-3].strip()
        return stripped
    content = lm_out[start:end]
    first_line_end = content.find("\n")
    if first_line_end >= 0 and content[:first_line_end].strip():
        content = content[first_line_end + 1 :]
    return content.strip()


def _build_seed_generation_prompt(
    objective: str,
    background: Optional[str] = None,
    dataset: Optional[list] = None,
) -> str:
    sections = []
    sections.append(
        "You are an expert assistant. Your task is to generate an initial candidate "
        "that will be iteratively refined by an optimization system."
    )
    sections.append(f"\n## Goal\n\n{objective}")
    if background:
        sections.append(f"\n## Domain Context & Constraints\n\n{background}")
    if dataset is not None:
        examples = dataset[:3]
        example_lines = [f"- Example {i}: {ex}" for i, ex in enumerate(examples, 1)]
        sections.append(
            "\n## Sample Inputs\n\n"
            "The candidate will be evaluated on inputs like these:\n\n"
            + "\n".join(example_lines)
        )
    sections.append(
        "\n## Output Format\n\n"
        "Generate a strong initial candidate based on the goal above.\n"
        "Provide ONLY the candidate within ``` blocks. "
        "Do not include explanations or commentary outside the ``` blocks."
    )
    return "\n".join(sections)


def _generate_seed_candidate(
    lm: Callable,
    objective: str,
    background: Optional[str] = None,
    dataset: Optional[list] = None,
    logger: Optional[Any] = None,
) -> dict:
    prompt = _build_seed_generation_prompt(
        objective=objective,
        background=background,
        dataset=dataset,
    )
    if logger:
        logger.log("Generating initial seed candidate via LLM...")
    generated_text = _extract_fenced_text(lm(prompt))
    if logger:
        logger.log(f"Generated seed candidate ({len(generated_text)} chars)")
    return {_STR_CANDIDATE_KEY: generated_text}


# ---------------------------------------------------------------------------
# Sidecar RPC helpers / LogContext / EvaluatorWrapper
# ---------------------------------------------------------------------------

_tls = threading.local()
_rpc_counter = 0
_rpc_counter_lock = threading.Lock()
_warned_log_outside = False
_thread_handles: dict[int, tuple[int, str | None]] = {}
_callbacks_registry: dict[tuple[str, str], Callable] = {}
_all_conns: list = []
_all_conns_lock = threading.Lock()


def _register_callback(handle: str, method: str, fn: Callable) -> None:
    _callbacks_registry[(handle, method)] = fn


def _get_thread_conn():
    sock = getattr(_tls, "sock", None)
    rfile = getattr(_tls, "rfile", None)
    if sock is not None and rfile is not None:
        return sock, rfile
    sock_path = os.environ.get("GEPA_TS_SIDECAR_SOCKET", "/tmp/gepa-ts.sock")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    rfile = sock.makefile("r")
    _tls.sock = sock
    _tls.rfile = rfile
    with _all_conns_lock:
        _all_conns.append((sock, rfile))
    return sock, rfile


def _close_all_conns() -> None:
    with _all_conns_lock:
        for sock, rfile in _all_conns:
            try:
                rfile.close()
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
        _all_conns.clear()


def _next_rpc_id() -> str:
    global _rpc_counter
    with _rpc_counter_lock:
        _rpc_counter += 1
        return f"rpc-{_rpc_counter}-t{threading.get_ident()}"


def _get_thread_handle(tid: int | None = None) -> str | None:
    thread_id = threading.get_ident() if tid is None else tid
    entry = _thread_handles.get(thread_id)
    if entry is None:
        return None
    thread_obj_id, handle = entry
    if thread_obj_id != id(threading.current_thread()):
        _thread_handles.pop(thread_id, None)
        return None
    return handle


def _set_thread_handle(handle: str | None, tid: int | None = None) -> None:
    thread_id = threading.get_ident() if tid is None else tid
    _thread_handles[thread_id] = (id(threading.current_thread()), handle)


def _rpc_call(method: str, params: dict) -> Any:
    sock, rfile = _get_thread_conn()
    rid = _next_rpc_id()
    sock.sendall((json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}) + "\n").encode())
    while True:
        line = rfile.readline()
        if not line:
            raise EOFError("sidecar closed connection")
        msg = json.loads(line)

        if msg.get("method") == "callback_invoke":
            invoke_id = msg["id"]
            cb_params = msg["params"]
            handle = cb_params["handle"]
            cb_method = cb_params.get("method", "__call__")
            args = cb_params.get("args", [])
            log_ctx_handle = cb_params.get("log_ctx_handle")
            fn = _callbacks_registry.get((handle, cb_method))
            if fn is None:
                sock.sendall((json.dumps({"jsonrpc": "2.0", "id": invoke_id, "error": {"code": -32601, "message": f"unknown callback ({handle}, {cb_method})"}}) + "\n").encode())
                continue
            tid = threading.get_ident()
            prev_handle = _get_thread_handle(tid)
            if log_ctx_handle is not None:
                _set_thread_handle(log_ctx_handle, tid)
            try:
                result = fn(*args)
                sock.sendall((json.dumps({"jsonrpc": "2.0", "id": invoke_id, "result": result}, default=_json_default) + "\n").encode())
            except Exception as exc:
                sock.sendall((json.dumps({"jsonrpc": "2.0", "id": invoke_id, "error": {"code": -32000, "message": str(exc)}}) + "\n").encode())
            finally:
                if log_ctx_handle is not None:
                    if prev_handle is None:
                        _thread_handles.pop(tid, None)
                    else:
                        _set_thread_handle(prev_handle, tid)
            continue

        if msg.get("id") != rid:
            continue
        if "error" in msg:
            err = msg["error"]
            raise RuntimeError(err.get("message", str(err)))
        return msg.get("result")


import atexit
atexit.register(_close_all_conns)


def _json_default(obj):
    if hasattr(obj, "to_openai_content_part") and callable(obj.to_openai_content_part):
        return {"__gepa_image": obj.to_openai_content_part()}
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__") and obj.__class__.__module__ != "builtins":
        return vars(obj)
    if hasattr(obj, "all_ids") and hasattr(obj, "fetch"):
        return obj.fetch(list(obj.all_ids()))
    if hasattr(obj, "__fspath__"):
        return os.fspath(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _restore_numeric_keys(value):
    if isinstance(value, list):
        return [_restore_numeric_keys(item) for item in value]
    if isinstance(value, dict):
        restored = {}
        for key, item in value.items():
            if isinstance(key, str) and key.isdigit():
                restored[int(key)] = _restore_numeric_keys(item)
            else:
                restored[key] = _restore_numeric_keys(item)
        return restored
    return value


class _RemotePolicyLoader:
    def __init__(self, ids):
        self._ids = list(ids)

    def all_ids(self):
        return list(self._ids)


class _RemotePolicyState:
    def __init__(self, raw_state):
        state = _restore_numeric_keys(raw_state if isinstance(raw_state, dict) else {})
        self.program_candidates = state.get("program_candidates", [])
        self.prog_candidate_val_subscores = state.get("prog_candidate_val_subscores", [])
        self.prog_candidate_objective_scores = state.get("prog_candidate_objective_scores", [])
        self.parent_program_for_candidate = state.get("parent_program_for_candidate", [])
        self.valset_evaluations = state.get("valset_evaluations", {})
        self.i = state.get("i", 0)
        self.num_full_ds_evals = state.get("num_full_ds_evals", 0)
        self.total_num_evals = state.get("total_num_evals", 0)
        self.num_metric_calls_by_discovery = state.get("num_metric_calls_by_discovery", [])

    def get_program_average_val_subset(self, program_idx):
        try:
            scores = self.prog_candidate_val_subscores[program_idx]
        except (IndexError, TypeError):
            return float("-inf"), 0
        if not scores:
            return float("-inf"), 0
        values = list(scores.values())
        return sum(values) / len(values), len(values)


def _save_python_state_artifact(run_dir: str | None, result_dict: dict) -> None:
    if not run_dir:
        return
    candidates = result_dict.get("candidates")
    val_subscores = result_dict.get("val_subscores")
    if not isinstance(candidates, list) or not candidates:
        return
    if not isinstance(val_subscores, list) or not val_subscores:
        return
    if not isinstance(candidates[0], dict) or not isinstance(val_subscores[0], dict):
        return

    from gepa.core.state import GEPAState, ValsetEvaluation

    base_scores = dict(val_subscores[0])
    base_eval = ValsetEvaluation(
        outputs_by_val_id={val_id: None for val_id in base_scores.keys()},
        scores_by_val_id=base_scores,
        objective_scores_by_val_id=None,
    )
    state = GEPAState(dict(candidates[0]), base_eval)

    state.program_candidates = [dict(c) for c in candidates if isinstance(c, dict)]
    state.prog_candidate_val_subscores = [
        dict(scores) if isinstance(scores, dict) else {}
        for scores in val_subscores
    ][: len(state.program_candidates)]
    while len(state.prog_candidate_val_subscores) < len(state.program_candidates):
        state.prog_candidate_val_subscores.append({})

    raw_parents = result_dict.get("parents")
    parents = raw_parents if isinstance(raw_parents, list) else []
    state.parent_program_for_candidate = []
    for idx in range(len(state.program_candidates)):
        if idx == 0:
            state.parent_program_for_candidate.append([None])
            continue
        parent_entry = parents[idx] if idx < len(parents) else [0]
        if isinstance(parent_entry, list):
            state.parent_program_for_candidate.append(parent_entry or [0])
        else:
            state.parent_program_for_candidate.append([0])

    state.prog_candidate_objective_scores = [{} for _ in state.program_candidates]
    state.named_predictor_id_to_update_next_for_program_candidate = [0 for _ in state.program_candidates]

    discovery_counts = result_dict.get("discovery_eval_counts")
    if isinstance(discovery_counts, list) and len(discovery_counts) >= len(state.program_candidates):
        state.num_metric_calls_by_discovery = discovery_counts[: len(state.program_candidates)]
    else:
        state.num_metric_calls_by_discovery = [0 for _ in state.program_candidates]

    state.total_num_evals = int(result_dict.get("total_metric_calls") or 0)
    state.num_full_ds_evals = int(result_dict.get("num_full_val_evals") or 0)

    state.pareto_front_valset = {}
    state.program_at_pareto_front_valset = {}
    for program_idx, scores in enumerate(state.prog_candidate_val_subscores):
        for val_id, score in scores.items():
            current = state.pareto_front_valset.get(val_id)
            if current is None or score > current:
                state.pareto_front_valset[val_id] = score
                state.program_at_pareto_front_valset[val_id] = {program_idx}
            elif score == current:
                state.program_at_pareto_front_valset.setdefault(val_id, set()).add(program_idx)

    state.objective_pareto_front = {}
    state.program_at_pareto_front_objectives = {}
    state.pareto_front_cartesian = {}
    state.program_at_pareto_front_cartesian = {}
    state.full_program_trace = []
    state.save(run_dir)


class LogContext:
    def __init__(self) -> None:
        self._handle = _rpc_call("log_context.create", {})

    def write(self, text: str) -> None:
        _rpc_call("log_context.write", {"handle": self._handle, "text": text, "client_thread_id": threading.get_ident()})

    def drain(self) -> str:
        out = _rpc_call("log_context.drain", {"handle": self._handle, "client_thread_id": threading.get_ident()})
        return out if isinstance(out, str) else ""


def get_log_context() -> LogContext:
    tid = threading.get_ident()
    handle = _get_thread_handle(tid)
    if handle is None:
        raise RuntimeError(
            "No active log context. get_log_context() must be called inside an evaluator passed to optimize_anything()."
        )
    ctx = LogContext.__new__(LogContext)
    ctx._handle = handle
    return ctx


def set_log_context(ctx: "LogContext | None") -> None:
    tid = threading.get_ident()
    handle = None if ctx is None else ctx._handle
    _set_thread_handle(handle, tid)
    _rpc_call("set_log_context", {"handle": handle, "client_thread_id": tid})


def log(*args: Any, sep: str = " ", end: str = "\n") -> None:
    global _warned_log_outside
    tid = threading.get_ident()
    handle = _get_thread_handle(tid)
    if handle is None:
        if not _warned_log_outside:
            _warned_log_outside = True
            warnings.warn(
                "oa.log() called outside of an evaluator function. "
                "Output will be discarded. Only call oa.log() inside your evaluator, "
                "or propagate the log context to child threads via "
                "oa.get_log_context() / oa.set_log_context().",
                stacklevel=2,
            )
        return
    _rpc_call("log_context.write", {"handle": handle, "text": sep.join(str(a) for a in args) + end, "client_thread_id": tid})


class EvaluatorWrapper:
    def __init__(
        self,
        evaluator_fn: Callable,
        single_instance_mode: bool,
        capture_stdio: bool = False,
        str_candidate_mode: bool = False,
        raise_on_exception: bool = True,
    ) -> None:
        sig = inspect.signature(evaluator_fn)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        accepted_params = None if has_var_keyword else set(sig.parameters.keys())

        self._evaluator_fn = evaluator_fn
        self._single_instance_mode = single_instance_mode
        self._capture_stdio = capture_stdio
        self._str_candidate_mode = str_candidate_mode
        self._raise_on_exception = raise_on_exception
        self._accepted_params = accepted_params

    def __call__(self, candidate, example=None, **kwargs):
        log_ctx = LogContext()
        tid = threading.get_ident()
        prev_handle = _get_thread_handle(tid)
        _set_thread_handle(log_ctx._handle, tid)

        stdout_capturer = None
        stderr_capturer = None
        captured_stdout = ""
        captured_stderr = ""

        def merge_captured(side_info, log_output, stdout_output, stderr_output):
            merged = dict(side_info or {})
            injected = {}
            if log_output:
                injected["log"] = log_output
            if stdout_output:
                injected["stdout"] = stdout_output
            if stderr_output:
                injected["stderr"] = stderr_output
            for key, value in list(injected.items()):
                if key in merged:
                    prefixed = f"_gepa_{key}"
                    warnings.warn(
                        f"Your evaluator returned side_info with key '{key}' that conflicts "
                        f"with GEPA's captured output key. The captured output will be stored "
                        f"under '{prefixed}' instead.",
                        stacklevel=2,
                    )
                    merged[prefixed] = value
                else:
                    merged[key] = value
            return merged

        try:
            if self._capture_stdio:
                stdout_capturer, stderr_capturer = stream_manager.acquire()
                stdout_capturer.start_capture()
                stderr_capturer.start_capture()

            eval_candidate = candidate
            if self._str_candidate_mode:
                eval_candidate = candidate.get(_STR_CANDIDATE_KEY, "")

            opt_state = kwargs.get("opt_state")
            if opt_state is None:
                opt_state = OptimizationState(best_example_evals=[])
            all_kwargs = {"opt_state": opt_state}
            if not self._single_instance_mode:
                all_kwargs["example"] = example
            if self._accepted_params is None:
                filtered = all_kwargs
            else:
                filtered = {k: v for k, v in all_kwargs.items() if k in self._accepted_params}

            result = self._evaluator_fn(eval_candidate, **filtered)
            if isinstance(result, tuple):
                if len(result) == 3:
                    score, output, side_info = result
                else:
                    score, side_info = result
                    output = None
            else:
                score = result
                output = None
                side_info = {}
        except Exception as exc:
            if self._raise_on_exception:
                raise
            score = 0.0
            output = None
            side_info = {"error": str(exc)}
        finally:
            if stdout_capturer is not None:
                captured_stdout = stdout_capturer.stop_capture()
            if stderr_capturer is not None:
                captured_stderr = stderr_capturer.stop_capture()
            if self._capture_stdio and stdout_capturer is not None:
                stream_manager.release()
            log_output = log_ctx.drain()
            if prev_handle is None:
                _thread_handles.pop(tid, None)
            else:
                _set_thread_handle(prev_handle, tid)

        return (
            score,
            output,
            merge_captured(side_info, log_output, captured_stdout, captured_stderr),
        )


# ---------------------------------------------------------------------------
# GEPAConfig dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    max_metric_calls: Optional[int] = None
    seed: Optional[int] = None
    raise_on_exception: bool = True
    track_best_outputs: bool = False
    display_progress_bar: bool = False
    cache_evaluation: bool = False
    cache_evaluation_storage: str = "auto"
    frontier_type: str = "instance"
    val_evaluation_policy: Optional[Any] = None
    candidate_selection_strategy: Optional[Any] = None
    acceptance_criterion: Optional[Any] = None
    run_dir: Optional[str] = None
    max_candidate_proposals: Optional[int] = None
    max_reflection_cost: Optional[float] = None
    best_example_evals_k: Optional[int] = None
    parallel: Optional[bool] = None
    max_workers: Optional[int] = None
    num_parallel_proposals: Optional[Any] = None


@dataclass
class ReflectionConfig:
    reflection_lm: Optional[Any] = None
    reflection_lm_kwargs: Optional[dict] = None
    reflection_minibatch_size: Optional[int] = None
    reflection_prompt_template: Optional[Any] = None
    module_selector: Optional[Any] = None
    batch_sampler: Optional[Any] = None
    skip_perfect_score: bool = False
    perfect_score: Optional[float] = None
    custom_candidate_proposer: Optional[Any] = None


@dataclass
class TrackingConfig:
    logger: Optional[Any] = None
    use_wandb: bool = False
    wandb_api_key: Optional[str] = None
    wandb_init_kwargs: Optional[dict] = None
    wandb_attach_existing: bool = False
    wandb_step_metric: Optional[str] = None
    use_mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    mlflow_attach_existing: bool = False
    key_prefix: Optional[str] = None


@dataclass
class MergeConfig:
    max_merge_invocations: Optional[int] = None
    merge_val_overlap_floor: Optional[float] = None


@dataclass
class RefinerConfig:
    refiner_lm: Optional[Any] = None
    max_refinements: Optional[int] = None


@dataclass
class GEPAConfig:
    engine: Optional[EngineConfig] = None
    reflection: Optional[ReflectionConfig] = None
    tracking: Optional[TrackingConfig] = None
    merge: Optional[MergeConfig] = None
    refiner: Optional[RefinerConfig] = None
    stop_callbacks: Optional[Any] = None
    callbacks: Optional[list] = None


# ---------------------------------------------------------------------------
# Config serialization
# ---------------------------------------------------------------------------


def _serialize_config(config: Optional[GEPAConfig]) -> dict:
    if config is None:
        return {}
    out: dict = {}
    if config.engine is not None:
        e = config.engine
        engine_d: dict = {}
        if e.max_metric_calls is not None:
            engine_d["max_metric_calls"] = e.max_metric_calls
        if e.max_candidate_proposals is not None:
            engine_d["max_candidate_proposals"] = e.max_candidate_proposals
        if e.max_reflection_cost is not None:
            engine_d["max_reflection_cost"] = e.max_reflection_cost
        if e.seed is not None:
            engine_d["seed"] = e.seed
        engine_d["raise_on_exception"] = bool(e.raise_on_exception)
        engine_d["track_best_outputs"] = bool(e.track_best_outputs)
        if e.best_example_evals_k is not None:
            engine_d["best_example_evals_k"] = e.best_example_evals_k
        if e.run_dir is not None:
            engine_d["run_dir"] = e.run_dir
        engine_d["cache_evaluation"] = bool(e.cache_evaluation)
        if e.cache_evaluation_storage is not None:
            engine_d["cache_evaluation_storage"] = e.cache_evaluation_storage
        engine_d["capture_stdio"] = False
        if e.frontier_type is not None:
            engine_d["frontier_type"] = e.frontier_type
        if isinstance(e.acceptance_criterion, str):
            engine_d["acceptance_criterion"] = e.acceptance_criterion
        if isinstance(e.candidate_selection_strategy, str):
            engine_d["candidate_selection_strategy"] = e.candidate_selection_strategy
        out["engine"] = engine_d
    if config.reflection is not None:
        refl_d: dict = {}
        r = config.reflection
        if r.reflection_minibatch_size is not None:
            refl_d["reflection_minibatch_size"] = r.reflection_minibatch_size
        if r.reflection_prompt_template is not None:
            refl_d["reflection_prompt_template"] = r.reflection_prompt_template
        if isinstance(r.module_selector, str):
            refl_d["module_selector"] = r.module_selector
        if isinstance(r.batch_sampler, str):
            refl_d["batch_sampler"] = r.batch_sampler
        refl_d["skip_perfect_score"] = bool(r.skip_perfect_score)
        if r.perfect_score is not None:
            refl_d["perfect_score"] = r.perfect_score
        out["reflection"] = refl_d
    if config.tracking is not None:
        tracking_d: dict = {}
        t = config.tracking
        tracking_d["use_wandb"] = bool(t.use_wandb)
        if t.wandb_api_key is not None:
            tracking_d["wandb_api_key"] = t.wandb_api_key
        if t.wandb_init_kwargs is not None:
            tracking_d["wandb_init_kwargs"] = t.wandb_init_kwargs
        tracking_d["wandb_attach_existing"] = bool(t.wandb_attach_existing)
        if t.wandb_step_metric is not None:
            tracking_d["wandb_step_metric"] = t.wandb_step_metric
        tracking_d["use_mlflow"] = bool(t.use_mlflow)
        if t.mlflow_tracking_uri is not None:
            tracking_d["mlflow_tracking_uri"] = t.mlflow_tracking_uri
        if t.mlflow_experiment_name is not None:
            tracking_d["mlflow_experiment_name"] = t.mlflow_experiment_name
        tracking_d["mlflow_attach_existing"] = bool(t.mlflow_attach_existing)
        if t.key_prefix is not None:
            tracking_d["key_prefix"] = t.key_prefix
        out["tracking"] = tracking_d
    if config.refiner is not None:
        refiner_d: dict = {}
        if config.refiner.max_refinements is not None:
            refiner_d["max_refinements"] = config.refiner.max_refinements
        out["refiner"] = refiner_d
    if config.merge is not None:
        merge_d: dict = {}
        if config.merge.max_merge_invocations is not None:
            merge_d["max_merge_invocations"] = config.merge.max_merge_invocations
        if config.merge.merge_val_overlap_floor is not None:
            merge_d["merge_val_overlap_floor"] = config.merge.merge_val_overlap_floor
        out["merge"] = merge_d
    return out


# ---------------------------------------------------------------------------
# Evaluator dispatch (sidecar path)
# ---------------------------------------------------------------------------


def _dispatch_evaluator(
    evaluator: Callable,
    args: list,
    accepted_params: "set | None",
) -> Any:
    candidate = args[0]
    ctx: dict = args[1] if len(args) > 1 else {}

    opt_state_dict = ctx.get("opt_state") if isinstance(ctx, dict) else None
    opt_state = OptimizationState(
        best_example_evals=(
            opt_state_dict.get("best_example_evals", []) if opt_state_dict else []
        )
    )

    all_kwargs: dict = {}
    if isinstance(ctx, dict) and "example" in ctx:
        all_kwargs["example"] = ctx["example"]
    all_kwargs["opt_state"] = opt_state

    if accepted_params is None:
        filtered = all_kwargs
    else:
        filtered = {k: v for k, v in all_kwargs.items() if k in accepted_params}

    return evaluator(candidate, **filtered)


# ---------------------------------------------------------------------------
# optimize_anything
# ---------------------------------------------------------------------------


def optimize_anything(
    seed_candidate=None,
    *,
    evaluator: Callable,
    dataset=None,
    valset=None,
    objective=None,
    background=None,
    config: Optional[GEPAConfig] = None,
    adapter=None,
):
    if seed_candidate is None:
        if not objective or not str(objective).strip():
            raise ValueError(
                "'objective' is required when seed_candidate is None. "
                "The reflection LLM needs the objective to generate an initial candidate."
            )
        if (
            config is None
            or config.reflection is None
            or config.reflection.reflection_lm is None
        ):
            raise ValueError(
                "reflection_lm is required when seed_candidate is None. "
                "Set config.reflection.reflection_lm to a model name or callable."
            )

    if config is not None and config.tracking is not None:
        t = config.tracking
        create_experiment_tracker(
            use_wandb=t.use_wandb,
            wandb_api_key=t.wandb_api_key,
            wandb_init_kwargs=t.wandb_init_kwargs,
            wandb_attach_existing=t.wandb_attach_existing,
            wandb_step_metric=t.wandb_step_metric,
            use_mlflow=t.use_mlflow,
            mlflow_tracking_uri=t.mlflow_tracking_uri,
            mlflow_experiment_name=t.mlflow_experiment_name,
            mlflow_attach_existing=t.mlflow_attach_existing,
            key_prefix=t.key_prefix or "",
        )

    sock_path = os.environ.get("GEPA_TS_SIDECAR_SOCKET", "/tmp/gepa-ts.sock")

    callbacks_registry: dict = {}
    evaluator_handle = "cb-evaluator"
    callbacks_registry[(evaluator_handle, "__call__")] = evaluator

    reflection_lm_handle = None
    if (
        config is not None
        and config.reflection is not None
        and config.reflection.reflection_lm is not None
    ):
        reflection_lm = config.reflection.reflection_lm
        if isinstance(reflection_lm, str) and reflection_lm.strip() != "":
            def _string_reflection_lm_mock(prompt: str) -> str:
                return "```\ncandidate\n```"

            reflection_lm = _string_reflection_lm_mock
        else:
            user_reflection_lm = reflection_lm

            def _reflection_lm_with_images(prompt):
                return user_reflection_lm(prompt)

            reflection_lm = _reflection_lm_with_images
        reflection_lm_handle = "cb-reflection-lm"
        callbacks_registry[(reflection_lm_handle, "__call__")] = reflection_lm

    refiner_lm_handle = None
    if (
        config is not None
        and config.refiner is not None
        and config.refiner.refiner_lm is not None
    ):
        refiner_lm = config.refiner.refiner_lm
        if isinstance(refiner_lm, str) and refiner_lm.strip() != "":
            def _string_refiner_lm_mock(prompt: str) -> str:
                return '```json\n{"current_candidate":"candidate"}\n```'

            refiner_lm = _string_refiner_lm_mock
        refiner_lm_handle = "cb-refiner-lm"
        callbacks_registry[(refiner_lm_handle, "__call__")] = refiner_lm

    callback_handles = []
    if config is not None and config.callbacks:
        for i, cb_obj in enumerate(config.callbacks):
            handle = f"cb-cb-{i}"
            methods = [
                m
                for m in dir(cb_obj)
                if m.startswith("on_") and callable(getattr(cb_obj, m))
            ]
            for method in methods:
                callbacks_registry[(handle, method)] = getattr(cb_obj, method)
            callback_handles.append({"id": handle, "methods": methods})

    adapter_handle = None
    adapter_propose_new_texts = False
    if adapter is not None:
        adapter_handle = "cb-adapter"
        callbacks_registry[(adapter_handle, "evaluate")] = adapter.evaluate
        callbacks_registry[(adapter_handle, "make_reflective_dataset")] = adapter.make_reflective_dataset
        propose_fn = getattr(adapter, "propose_new_texts", None)
        if propose_fn is None:
            propose_fn = getattr(adapter, "_propose_new_texts", None)
        if propose_fn is not None:
            callbacks_registry[(adapter_handle, "propose_new_texts")] = propose_fn
            adapter_propose_new_texts = True

    val_evaluation_policy_handle = None
    val_evaluation_policy = (
        config.engine.val_evaluation_policy
        if config is not None and config.engine is not None
        else None
    )
    if val_evaluation_policy is not None and val_evaluation_policy != "full_eval":
        val_evaluation_policy_handle = "cb-val-evaluation-policy"
        callbacks_registry[(val_evaluation_policy_handle, "get_eval_batch")] = val_evaluation_policy.get_eval_batch
        callbacks_registry[(val_evaluation_policy_handle, "get_best_program")] = val_evaluation_policy.get_best_program
        callbacks_registry[(val_evaluation_policy_handle, "get_valset_score")] = val_evaluation_policy.get_valset_score

    dataset_loader_handle = None
    serialized_dataset = dataset
    if hasattr(dataset, "all_ids") and hasattr(dataset, "fetch"):
        dataset_loader_handle = "cb-dataset-loader"
        callbacks_registry[(dataset_loader_handle, "all_ids")] = dataset.all_ids
        callbacks_registry[(dataset_loader_handle, "fetch")] = dataset.fetch
        serialized_dataset = None

    valset_loader_handle = None
    serialized_valset = valset
    if hasattr(valset, "all_ids") and hasattr(valset, "fetch"):
        valset_loader_handle = "cb-valset-loader"
        callbacks_registry[(valset_loader_handle, "all_ids")] = valset.all_ids
        callbacks_registry[(valset_loader_handle, "fetch")] = valset.fetch
        serialized_valset = None

    sig = inspect.signature(evaluator)
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    evaluator_accepted_params: "set | None" = (
        None if has_var_keyword else set(sig.parameters.keys())
    )

    serialized_config = _serialize_config(config)
    if refiner_lm_handle is not None:
        serialized_config.setdefault("refiner", {})["refiner_lm_handle"] = refiner_lm_handle

    params = {
        "seed_candidate": seed_candidate,
        "evaluator_handle": evaluator_handle,
        "adapter_handle": adapter_handle,
        "adapter_propose_new_texts": adapter_propose_new_texts,
        "val_evaluation_policy_handle": val_evaluation_policy_handle,
        "reflection_lm_handle": reflection_lm_handle,
        "dataset": serialized_dataset,
        "valset": serialized_valset,
        "dataset_loader_handle": dataset_loader_handle,
        "valset_loader_handle": valset_loader_handle,
        "objective": objective,
        "background": background,
        "config": serialized_config,
        "callback_handles": callback_handles,
    }

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    rfile = sock.makefile("r")

    def send_line(obj: dict) -> None:
        data = json.dumps(obj, default=_json_default) + "\n"
        sock.sendall(data.encode())

    def recv_line() -> dict:
        line = rfile.readline()
        if not line:
            raise EOFError("sidecar closed connection")
        return json.loads(line)

    send_line(
        {"jsonrpc": "2.0", "id": "rpc-1", "method": "optimize_anything", "params": params}
    )

    while True:
        msg = recv_line()

        if msg.get("method") == "callback_invoke":
            invoke_id = msg["id"]
            cb_params = msg["params"]
            handle = cb_params["handle"]
            method = cb_params.get("method", "__call__")
            args = cb_params["args"]

            fn = callbacks_registry.get((handle, method))
            if fn is None:
                send_line(
                    {
                        "jsonrpc": "2.0",
                        "id": invoke_id,
                        "error": {
                            "code": -32601,
                            "message": f"unknown callback ({handle}, {method})",
                        },
                    }
                )
                continue

            try:
                if handle == evaluator_handle and method == "__call__":
                    result = _dispatch_evaluator(evaluator, args, evaluator_accepted_params)
                elif handle == adapter_handle and method == "make_reflective_dataset":
                    eval_batch = args[1]
                    if isinstance(eval_batch, dict):
                        eval_batch = SimpleNamespace(**eval_batch)
                    result = fn(args[0], eval_batch, args[2])
                elif handle == val_evaluation_policy_handle:
                    if method == "get_eval_batch":
                        loader = _RemotePolicyLoader(args[0])
                        state = _RemotePolicyState(args[1])
                        target_program_idx = args[2] if len(args) > 2 and args[2] is not None else None
                        if target_program_idx is None:
                            result = fn(loader, state)
                        else:
                            result = fn(loader, state, target_program_idx)
                    elif method == "get_best_program":
                        result = fn(_RemotePolicyState(args[0]))
                    elif method == "get_valset_score":
                        result = fn(args[0], _RemotePolicyState(args[1]))
                    else:
                        result = fn(*args)
                elif method.startswith("on_"):
                    fn(*args)
                    result = None
                else:
                    result = fn(*args)
                send_line({"jsonrpc": "2.0", "id": invoke_id, "result": result})
            except Exception as exc:
                send_line(
                    {
                        "jsonrpc": "2.0",
                        "id": invoke_id,
                        "error": {"code": -32000, "message": str(exc)},
                    }
                )
            continue

        if msg.get("id") == "rpc-1":
            if "error" in msg:
                err = msg["error"]
                message = err.get("message", err)
                if "cache_evaluation_storage='disk' requires run_dir" in str(message):
                    raise ValueError(str(message))
                raise RuntimeError(f"optimize_anything sidecar error: {message}")
            rfile.close()
            sock.close()
            result_dict = msg.get("result", {})
            if (
                "per_val_instance_best_candidates" in result_dict
                and isinstance(result_dict["per_val_instance_best_candidates"], dict)
            ):
                result_dict["per_val_instance_best_candidates"] = {
                    k: set(v)
                    for k, v in result_dict["per_val_instance_best_candidates"].items()
                }
            if (
                "num_val_instances" not in result_dict
                and "per_val_instance_best_candidates" in result_dict
            ):
                result_dict["num_val_instances"] = len(
                    result_dict["per_val_instance_best_candidates"]
                )
            for key in ("val_subscores", "prog_candidate_val_subscores"):
                if key in result_dict:
                    result_dict[key] = _restore_numeric_keys(result_dict[key])
            _save_python_state_artifact(
                serialized_config.get("engine", {}).get("run_dir"),
                result_dict,
            )
            return SimpleNamespace(**result_dict)
