from pathlib import Path

_UPSTREAM_UTILS = Path("/tmp/gepa-upstream-ce51b50/src/gepa/utils")
if _UPSTREAM_UTILS.exists():
    __path__.append(str(_UPSTREAM_UTILS))

from .stop_condition import (  # noqa: E402
    CompositeStopper,
    FileStopper,
    MaxCandidateProposalsStopper,
    MaxMetricCallsStopper,
    MaxReflectionCostStopper,
    NoImprovementStopper,
    ScoreThresholdStopper,
    SignalStopper,
    StopperProtocol,
    TimeoutStopCondition,
)

__all__ = [
    "CompositeStopper",
    "FileStopper",
    "MaxCandidateProposalsStopper",
    "MaxMetricCallsStopper",
    "MaxReflectionCostStopper",
    "NoImprovementStopper",
    "ScoreThresholdStopper",
    "SignalStopper",
    "StopperProtocol",
    "TimeoutStopCondition",
]
