import json
import os
import socket
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional


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
        out["engine"] = engine_d
    if config.reflection is not None:
        refl_d: dict = {}
        r = config.reflection
        if r.reflection_minibatch_size is not None:
            refl_d["reflection_minibatch_size"] = r.reflection_minibatch_size
        out["reflection"] = refl_d
    return out


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

    callbacks: dict[str, Callable] = {}
    evaluator_handle = "cb-evaluator"
    callbacks[evaluator_handle] = evaluator

    reflection_lm_handle = None
    if config is not None and config.reflection is not None and config.reflection.reflection_lm is not None:
        reflection_lm_handle = "cb-reflection-lm"
        callbacks[reflection_lm_handle] = config.reflection.reflection_lm

    params = {
        "seed_candidate": seed_candidate,
        "evaluator_handle": evaluator_handle,
        "reflection_lm_handle": reflection_lm_handle,
        "dataset": dataset,
        "valset": valset,
        "objective": objective,
        "background": background,
        "config": _serialize_config(config),
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

    send_line({"jsonrpc": "2.0", "id": "rpc-1", "method": "optimize_anything", "params": params})

    while True:
        msg = recv_line()

        # callback_invoke from sidecar
        if msg.get("method") == "callback_invoke":
            invoke_id = msg["id"]
            cb_params = msg["params"]
            handle = cb_params["handle"]
            args = cb_params["args"]
            fn = callbacks.get(handle)
            if fn is None:
                send_line({"jsonrpc": "2.0", "id": invoke_id, "error": {"code": -32601, "message": f"unknown handle: {handle}"}})
                continue
            try:
                result = fn(*args)
                send_line({"jsonrpc": "2.0", "id": invoke_id, "result": result})
            except Exception as exc:
                send_line({"jsonrpc": "2.0", "id": invoke_id, "error": {"code": -32000, "message": str(exc)}})
            continue

        # response to our optimize_anything request
        if msg.get("id") == "rpc-1":
            if "error" in msg:
                err = msg["error"]
                raise RuntimeError(f"optimize_anything sidecar error: {err.get('message', err)}")
            rfile.close()
            sock.close()
            result_dict = msg.get("result", {})
            return SimpleNamespace(**result_dict)

        # unexpected message — skip
