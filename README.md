# @twaldin/gepa-ts

A TypeScript port of [gepa](https://github.com/gepa-ai/gepa) — Genetic-Pareto reflective text evolution for prompt and program optimization.

**Package version: v0.1.0.** Upstream pin: `ce51b50cd196b539c25fae99ad0e0255c23004a4`.

## Design constraints

- **Python parity target**: intentional snake_case API and upstream behavior as the spec. The acceptance gate runs selected, unmodified upstream pytests via a Python sidecar shim; it does not cover the entire upstream suite.
- **Bring your own LLM**: zero SDK dependencies. The reflection LM has type `(prompt: string | ChatMessage[]) => Promise<string>`; the current reflection path sends string prompts. Wrap whatever you use (Anthropic, OpenAI, OpenRouter, local).
- **Zero runtime deps**: no transitive supply-chain footprint.
- **Runtimes**: Node ≥20, Bun ≥1. Dual ESM + CJS exports. The current local build does not run directly under Deno: its emitted ESM imports bare `async_hooks`, which Deno rejects without Node-compatible resolution.

## v0.1 scope

`optimize_anything` with a BYO `evaluator`: genetic loop, reflective mutation, candidate evaluation, per-instance Pareto frontier maintenance, structured logging via `oa_log` / `LogContext` / `getLogContext` / `setLogContext` (AsyncLocalStorage-backed), and `EvaluatorWrapper` with `capture_stdio`. The exported engine and adapter building blocks expose `GEPAAdapter.evaluate` and `make_reflective_dataset` for lower-level integrations.

Excluded from v0.1: checkpointing, evaluation cache, multimodal, refiner, reflection cost tracking, seedless mode.

## Build and install from source

The public npm registry currently returns 404 for `@twaldin/gepa-ts`; use a local build rather than `npm install @twaldin/gepa-ts`.

```bash
git clone --branch main https://github.com/twaldin/gepa-ts.git
cd gepa-ts
bun install --frozen-lockfile
bun run build
```

Then, from your consuming project:

```bash
npm install /absolute/path/to/gepa-ts
```

## Quick start

This offline wiring example uses a fixed reflection response, not a provider call. Both candidates score equally, so the default strict-improvement rule keeps the seed.

```ts
import { optimize_anything } from '@twaldin/gepa-ts';

const result = await optimize_anything({
  seed_candidate: 'Answer the question concisely.',
  evaluator: (candidate) => {
    const prompt = String(candidate);
    const score = prompt.length < 100 ? 1.0 : 0.0;
    return [score, { len: prompt.length }];
  },
  config: {
    engine: { max_metric_calls: 20 },
    reflection: {
      reflection_lm: async (prompt) => {
        // Replace this fixed response with your LLM call.
        return 'Improved answer.';
      },
    },
  },
});

console.log(result.best_candidate);
```

## Acceptance gate

Each release must pass:

- `bun run typecheck` (strict TS)
- `bun run test` (vitest, full TS suite)
- `bash tests/run-pytest-shim.sh` — selected upstream gepa pytests via a unix-socket sidecar bridge: `test_optimize_anything_callbacks.py`, `test_evaluator_wrapper.py::TestOaLog`, `test_best_example_evals.py`, and `test_optimize.py`. This is a subset, not a full-suite parity claim.

Tamper validation of sidecar-backed calls (for example, making `LogContext.write` a no-op) checks that those paths exercise TS. The shim also performs Python-side validation and compatibility work; passing its selected tests does not establish full upstream parity.

## License

MIT.
