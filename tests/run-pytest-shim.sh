#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOCK="${GEPA_TS_SIDECAR_SOCKET:-/tmp/gepa-ts.sock}"
UPSTREAM="/tmp/gepa-upstream-ce51b50"

if [ ! -d "$UPSTREAM" ]; then
  git clone https://github.com/gepa-ai/gepa "$UPSTREAM"
fi
(cd "$UPSTREAM" && git checkout ce51b50cd196b539c25fae99ad0e0255c23004a4 2>/dev/null || true)

# Build sidecar
(cd "$ROOT" && npx --no-install tsup src/sidecar/main.ts --format cjs --out-dir dist/sidecar --silent || \
  npm exec --offline tsup -- src/sidecar/main.ts --format cjs --out-dir dist/sidecar --silent || \
  npx tsup src/sidecar/main.ts --format cjs --out-dir dist/sidecar --silent)

# Start sidecar
rm -f "$SOCK"
GEPA_TS_SIDECAR_SOCKET="$SOCK" node "$ROOT/dist/sidecar/main.cjs" &
SIDECAR_PID=$!
trap "kill $SIDECAR_PID 2>/dev/null || true; rm -f $SOCK" EXIT

# Wait for socket (up to 2s)
for _ in 1 2 3 4 5 6 7 8 9 10; do
  [ -S "$SOCK" ] && break
  sleep 0.2
done
[ -S "$SOCK" ] || { echo 'sidecar failed to bind'; exit 1; }

# Run pytest with shim shadowing real gepa
PYTHONPATH="$ROOT/tests/pytest-shim:${PYTHONPATH:-}" \
GEPA_TS_SIDECAR_SOCKET="$SOCK" \
  pytest "$UPSTREAM/tests/test_optimize_anything_callbacks.py::TestOptimizeAnythingCallbacks::test_no_callbacks_default" -v -s
