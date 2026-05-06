# @twaldin/gepa-ts

A 1-1 TypeScript port of [gepa](https://github.com/gepa-ai/gepa) — Genetic-Pareto reflective text evolution for prompt and program optimization.

**v0.1.0** — first published release. Upstream pin: `ce51b50cd196b539c25fae99ad0e0255c23004a4`.

## Design constraints

- **1-1 with Python**: snake_case API, behavior-equivalent. The acceptance gate is the upstream pytest suite running unmodified against this implementation via a Python sidecar shim.
- **Bring your own LLM**: zero SDK dependencies. The reflection LM is a single function: `(prompt: string) => Promise<string>`. Wrap whatever you use (Anthropic, OpenAI, OpenRouter, local).
- **Zero runtime deps**: no transitive supply-chain footprint.
- **Cross-runtime**: Node ≥20, Bun ≥1, Deno. Dual ESM + CJS exports.

## v0.1 scope

`optimize_anything` and its full transitive closure: genetic loop, reflective mutation, candidate evaluation, Pareto frontier maintenance, BYO `evaluator` and `make_reflective_dataset`, structured logging via `oa_log` / `LogContext` / `getLogContext` / `setLogContext` (AsyncLocalStorage-backed), and `EvaluatorWrapper` with `capture_stdio`.

Excluded from v0.1: checkpointing, evaluation cache, multimodal, refiner, reflection cost tracking, seedless mode.

## Install

```bash
npm install @twaldin/gepa-ts
# or: bun add @twaldin/gepa-ts
```

## Quick start

```ts
import { optimize_anything } from '@twaldin/gepa-ts';

const result = await optimize_anything({
  seed_candidate: 'Answer the question concisely.',
  evaluator: (candidate, ctx) => {
    const prompt = String(candidate);
    const score = prompt.length < 100 ? 1.0 : 0.0;
    return [score, { len: prompt.length }];
  },
  config: {
    engine: { max_metric_calls: 20 },
    reflection: {
      reflection_lm: async (prompt) => {
        // call your LLM here
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
- `bash tests/run-pytest-shim.sh` — upstream gepa pytests run unmodified against this implementation via a unix-socket sidecar bridge

Tamper-validated: no-op'ing core TS functions (e.g. `LogContext.write`) causes upstream pytests to fail, proving the bridge truly cross-validates the TypeScript port.

## License

MIT.
