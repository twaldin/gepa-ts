from pathlib import Path

_UPSTREAM_ADAPTERS = Path("/tmp/gepa-upstream-ce51b50/src/gepa/adapters")
if _UPSTREAM_ADAPTERS.exists():
    __path__.append(str(_UPSTREAM_ADAPTERS))
