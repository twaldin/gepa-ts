import sys
import importlib
from pathlib import Path

print('[gepa-ts-shim] active', file=sys.stderr)

_UPSTREAM_GEPA = Path("/tmp/gepa-upstream-ce51b50/src/gepa")
if _UPSTREAM_GEPA.exists():
    __path__.append(str(_UPSTREAM_GEPA))

try:
    examples = importlib.import_module("gepa.examples")
    importlib.import_module("gepa.examples.aime")
except Exception:
    examples = None

from gepa.optimize_anything import (
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
    TrackingConfig,
    MergeConfig,
    RefinerConfig,
    OptimizationState,
    EvaluatorWrapper,
    LogContext,
    log,
    get_log_context,
    set_log_context,
)
from gepa.api import optimize

__all__ = [
    'GEPAConfig',
    'EngineConfig',
    'ReflectionConfig',
    'TrackingConfig',
    'MergeConfig',
    'RefinerConfig',
    'OptimizationState',
    'EvaluatorWrapper',
    'LogContext',
    'log',
    'get_log_context',
    'set_log_context',
    'optimize',
]
