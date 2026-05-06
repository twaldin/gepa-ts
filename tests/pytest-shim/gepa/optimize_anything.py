import inspect
import io
import json
import os
import socket
import threading
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional

from gepa.utils.stdio_capture import ThreadLocalStreamCapture, stream_manager


# ---------------------------------------------------------------------------
# OptimizationState
# ---------------------------------------------------------------------------


@dataclass
class OptimizationState:
    best_example_evals: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# LogContext and oa.log()
# ---------------------------------------------------------------------------


class LogContext:
    def __init__(self) -> None:
        self._buffer = io.StringIO()
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        with self._lock:
            self._buffer.write(text)

    def drain(self) -> str:
        with self._lock:
            old = self._buffer
            text = old.getvalue()
            old.close()
            self._buffer = io.StringIO()
            return text


_log_tls = threading.local()


def _get_log_context() -> "LogContext | None":
    return getattr(_log_tls, "context", None)


def _set_log_context(ctx: "LogContext | None") -> None:
    _log_tls.context = ctx


def get_log_context() -> LogContext:
    ctx = _get_log_context()
    if ctx is None:
        raise RuntimeError(
            "No active log context. get_log_context() must be called inside an evaluator passed to optimize_anything()."
        )
    return ctx


def set_log_context(ctx: LogContext) -> None:
    _set_log_context(ctx)


def log(*args: Any, sep: str = " ", end: str = "\n") -> None:
    ctx = _get_log_context()
    if ctx is None:
        warnings.warn(
            "oa.log() called outside of an evaluator function. "
            "Output will be discarded. Only call oa.log() inside your evaluator, "
            "or propagate the log context to child threads via "
            "oa.get_log_context() / oa.set_log_context().",
            stacklevel=2,
        )
        return
    text = sep.join(str(a) for a in args) + end
    ctx.write(text)


# ---------------------------------------------------------------------------
# EvaluatorWrapper
# ---------------------------------------------------------------------------

_STR_CANDIDATE_KEY = "current_candidate"


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
        if has_var_keyword:
            accepted_params = None
        else:
            accepted_params = set(sig.parameters.keys())

        def _filter_kwargs(kwargs: dict) -> dict:
            if accepted_params is None:
                return kwargs
            return {k: v for k, v in kwargs.items() if k in accepted_params}

        def wrapped_evaluator(candidate, example=None, **kwargs):
            log_ctx = LogContext()
            _set_log_context(log_ctx)

            if single_instance_mode:
                all_kwargs = kwargs
            else:
                all_kwargs = {"example": example, **kwargs}

            filtered = _filter_kwargs(all_kwargs)

            eval_candidate = candidate
            if str_candidate_mode:
                eval_candidate = candidate[_STR_CANDIDATE_KEY]

            stdout_capturer: "ThreadLocalStreamCapture | None" = None
            stderr_capturer: "ThreadLocalStreamCapture | None" = None
            try:
                if capture_stdio:
                    stdout_capturer, stderr_capturer = stream_manager.acquire()
                    stdout_capturer.start_capture()
                    stderr_capturer.start_capture()

                result = evaluator_fn(eval_candidate, **filtered)
            except Exception as e:
                result = e
            finally:
                captured_stdout = stdout_capturer.stop_capture() if stdout_capturer else ""
                captured_stderr = stderr_capturer.stop_capture() if stderr_capturer else ""
                if capture_stdio and stdout_capturer is not None:
                    stream_manager.release()
                log_output = log_ctx.drain()
                _set_log_context(None)

            if isinstance(result, Exception):
                if raise_on_exception:
                    raise result
                fail_side_info: dict = {"error": str(result)}
                if log_output:
                    fail_side_info["log"] = log_output
                if captured_stdout:
                    fail_side_info["stdout"] = captured_stdout
                if captured_stderr:
                    fail_side_info["stderr"] = captured_stderr
                return 0.0, None, fail_side_info

            if isinstance(result, tuple):
                score, side_info = result
                side_info = dict(side_info) if side_info is not None else {}

                injected: dict = {}
                if log_output:
                    injected["log"] = log_output
                if captured_stdout:
                    injected["stdout"] = captured_stdout
                if captured_stderr:
                    injected["stderr"] = captured_stderr

                for key in list(injected):
                    if key in side_info:
                        prefixed = f"_gepa_{key}"
                        warnings.warn(
                            f"Your evaluator returned side_info with key '{key}' that conflicts "
                            f"with GEPA's captured output key. The captured output will be stored "
                            f"under '{prefixed}' instead.",
                            stacklevel=2,
                        )
                        injected[prefixed] = injected.pop(key)

                side_info.update(injected)
                return score, None, side_info
            else:
                score = result
                auto_side_info: dict = {}
                if captured_stdout:
                    auto_side_info["stdout"] = captured_stdout
                if captured_stderr:
                    auto_side_info["stderr"] = captured_stderr
                if log_output:
                    auto_side_info["log"] = log_output
                return score, None, auto_side_info

        self._wrapped = wrapped_evaluator

    def __call__(self, candidate, example=None, **kwargs):
        return self._wrapped(candidate, example=example, **kwargs)


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
        reflection_lm_handle = "cb-reflection-lm"
        callbacks_registry[(reflection_lm_handle, "__call__")] = config.reflection.reflection_lm

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
