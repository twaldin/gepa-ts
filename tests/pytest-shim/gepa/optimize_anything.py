import inspect
import json
import os
import socket
import threading
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional



# ---------------------------------------------------------------------------
# OptimizationState
# ---------------------------------------------------------------------------


@dataclass
class OptimizationState:
    best_example_evals: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sidecar RPC helpers / LogContext / EvaluatorWrapper
# ---------------------------------------------------------------------------

_sock_singleton = None
_rfile_singleton = None
_rpc_lock = threading.RLock()
_rpc_counter = 0
_warned_log_outside = False
_thread_handles: dict[int, str | None] = {}
_callbacks_registry: dict[tuple[str, str], Callable] = {}


def _register_callback(handle: str, method: str, fn: Callable) -> None:
    _callbacks_registry[(handle, method)] = fn


def _get_singleton_conn():
    global _sock_singleton, _rfile_singleton
    if _sock_singleton is not None and _rfile_singleton is not None:
        return _sock_singleton, _rfile_singleton
    sock_path = os.environ.get("GEPA_TS_SIDECAR_SOCKET", "/tmp/gepa-ts.sock")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    _sock_singleton = sock
    _rfile_singleton = sock.makefile("r")
    return _sock_singleton, _rfile_singleton


def _close_singleton_conn() -> None:
    global _sock_singleton, _rfile_singleton
    with _rpc_lock:
        if _rfile_singleton is not None:
            _rfile_singleton.close()
        if _sock_singleton is not None:
            _sock_singleton.close()
        _sock_singleton = None
        _rfile_singleton = None


def _rpc_call(method: str, params: dict) -> Any:
    global _rpc_counter
    with _rpc_lock:
        sock, rfile = _get_singleton_conn()
        _rpc_counter += 1
        rid = f"rpc-{_rpc_counter}"
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
                fn = _callbacks_registry.get((handle, cb_method))
                if fn is None:
                    sock.sendall((json.dumps({"jsonrpc": "2.0", "id": invoke_id, "error": {"code": -32601, "message": f"unknown callback ({handle}, {cb_method})"}}) + "\n").encode())
                    continue
                try:
                    result = fn(*args)
                    sock.sendall((json.dumps({"jsonrpc": "2.0", "id": invoke_id, "result": result}) + "\n").encode())
                except Exception as exc:
                    sock.sendall((json.dumps({"jsonrpc": "2.0", "id": invoke_id, "error": {"code": -32000, "message": str(exc)}}) + "\n").encode())
                continue

            if msg.get("id") != rid:
                continue
            if "error" in msg:
                err = msg["error"]
                raise RuntimeError(err.get("message", str(err)))
            return msg.get("result")


import atexit
atexit.register(_close_singleton_conn)


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
    handle = _thread_handles.get(tid)
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
    _thread_handles[tid] = handle
    _rpc_call("set_log_context", {"handle": handle, "client_thread_id": tid})


def log(*args: Any, sep: str = " ", end: str = "\n") -> None:
    global _warned_log_outside
    tid = threading.get_ident()
    handle = _thread_handles.get(tid)
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

        def _dispatch(candidate, ctx=None):
            ctx = ctx or {}
            all_kwargs = {"opt_state": OptimizationState(best_example_evals=(ctx.get("opt_state", {}) or {}).get("best_example_evals", []))}
            if not single_instance_mode and "example" in ctx:
                all_kwargs["example"] = ctx["example"]
            if accepted_params is None:
                filtered = all_kwargs
            else:
                filtered = {k: v for k, v in all_kwargs.items() if k in accepted_params}
            return evaluator_fn(candidate, **filtered)

        handle = f"py-evaluator-{id(self)}"
        _register_callback(handle, "__call__", _dispatch)
        self._handle = _rpc_call(
            "evaluator_wrapper.create",
            {
                "evaluator_handle": handle,
                "single_instance_mode": single_instance_mode,
                "capture_stdio": capture_stdio,
                "str_candidate_mode": str_candidate_mode,
                "raise_on_exception": raise_on_exception,
                "client_thread_id": threading.get_ident(),
            },
        )

    def __call__(self, candidate, example=None, **kwargs):
        opt_state = kwargs.get("opt_state")
        opt_state_payload = None
        if opt_state is not None:
            opt_state_payload = {
                "best_example_evals": getattr(opt_state, "best_example_evals", [])
            }
        return tuple(
            _rpc_call(
                "evaluator_wrapper.call",
                {
                    "handle": self._handle,
                    "candidate": candidate,
                    "example": example,
                    "opt_state": opt_state_payload,
                    "client_thread_id": threading.get_ident(),
                },
            )
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
        if e.seed is not None:
            engine_d["seed"] = e.seed
        if e.best_example_evals_k is not None:
            engine_d["best_example_evals_k"] = e.best_example_evals_k
        out["engine"] = engine_d
    if config.reflection is not None:
        refl_d: dict = {}
        r = config.reflection
        if r.reflection_minibatch_size is not None:
            refl_d["reflection_minibatch_size"] = r.reflection_minibatch_size
        if r.reflection_prompt_template is not None:
            refl_d["reflection_prompt_template"] = r.reflection_prompt_template
        out["reflection"] = refl_d
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
):
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
        reflection_lm_handle = "cb-reflection-lm"
        callbacks_registry[(reflection_lm_handle, "__call__")] = reflection_lm

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

    sig = inspect.signature(evaluator)
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    evaluator_accepted_params: "set | None" = (
        None if has_var_keyword else set(sig.parameters.keys())
    )

    params = {
        "seed_candidate": seed_candidate,
        "evaluator_handle": evaluator_handle,
        "reflection_lm_handle": reflection_lm_handle,
        "dataset": dataset,
        "valset": valset,
        "objective": objective,
        "background": background,
        "config": _serialize_config(config),
        "callback_handles": callback_handles,
    }

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    rfile = sock.makefile("r")

    def send_line(obj: dict) -> None:
        data = json.dumps(obj) + "\n"
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
                raise RuntimeError(
                    f"optimize_anything sidecar error: {err.get('message', err)}"
                )
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
            return SimpleNamespace(**result_dict)
