import sys

print('[gepa-ts-shim] active', file=sys.stderr)

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
