from types import SimpleNamespace
from pathlib import Path
import uuid


_active_stack = []
_tracking_uri = None
_experiment_name = "Default"
_experiments = {}
_runs = {}


def _tracking_path(uri=None):
    value = uri or _tracking_uri
    if isinstance(value, str) and value.startswith("file://"):
        return Path(value[7:])
    return None


def _ensure_tracking_dir(uri=None):
    path = _tracking_path(uri)
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)


def set_tracking_uri(uri):
    global _tracking_uri
    _tracking_uri = uri
    _ensure_tracking_dir(uri)


def get_tracking_uri():
    return _tracking_uri


def set_experiment(name):
    global _experiment_name
    _experiment_name = name
    _experiments.setdefault(name, SimpleNamespace(experiment_id=str(len(_experiments) + 1), name=name))
    _ensure_tracking_dir()
    return _experiments[name]


def active_run():
    return _active_stack[-1] if _active_stack else None


class _RunContext:
    def __init__(self, run):
        self._run = run

    def __enter__(self):
        return self._run

    def __exit__(self, exc_type, exc, tb):
        end_run()
        return False

    def __getattr__(self, name):
        return getattr(self._run, name)


def start_run(*args, **kwargs):
    _ensure_tracking_dir()
    if _experiment_name not in _experiments:
        set_experiment(_experiment_name)
    run_id = kwargs.get("run_id") or f"shim-{uuid.uuid4().hex}"
    run = SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, experiment_id=_experiments[_experiment_name].experiment_id),
        data=SimpleNamespace(metrics={}, params={}),
    )
    _runs[run_id] = run
    _active_stack.append(run)
    return _RunContext(run)


def end_run(*args, **kwargs):
    if _active_stack:
        _active_stack.pop()


def log_metric(key, value, step=None):
    run = active_run()
    if run is not None:
        run.data.metrics[key] = value


def log_metrics(metrics, step=None):
    for key, value in dict(metrics or {}).items():
        if isinstance(value, (int, float)):
            log_metric(key, float(value), step=step)


def log_param(key, value):
    run = active_run()
    if run is not None:
        run.data.params[key] = str(value)


def log_params(params):
    for key, value in dict(params or {}).items():
        log_param(key, value)


def log_table(*args, **kwargs):
    return None


def log_artifact(*args, **kwargs):
    return None


class MlflowClient:
    def __init__(self, tracking_uri=None, *args, **kwargs):
        self.tracking_uri = tracking_uri or _tracking_uri
        _ensure_tracking_dir(self.tracking_uri)

    def get_experiment_by_name(self, name):
        return _experiments.get(name)

    def search_runs(self, experiment_ids=None):
        wanted = {str(x) for x in (experiment_ids or [])}
        runs = list(_runs.values())
        if wanted:
            runs = [run for run in runs if str(run.info.experiment_id) in wanted]
        return list(reversed(runs))

    def get_run(self, run_id):
        return _runs[run_id]

    def log_param(self, run_id, key, value):
        _runs[run_id].data.params[key] = str(value)

    def log_metric(self, run_id, key, value, step=None):
        _runs[run_id].data.metrics[key] = value

    def log_artifact(self, *args, **kwargs):
        return None
