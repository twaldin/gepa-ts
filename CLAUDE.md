# gepa-ts

A TypeScript port of [gepa](https://github.com/gepa-ai/gepa) — Genetic-Pareto reflective text evolution for prompt and program optimization. Candidate prompts are mutated by an LLM that reflects on prior successes and failures, and a Pareto frontier over per-instance scores keeps a diverse pool of strong candidates alive across iterations instead of collapsing to a single greedy winner.

Package name: `@twaldin/gepa-ts`, version v0.1.0. Use the source-build installation in [README.md](README.md); the public npm registry currently returns 404 for this package. v0.1 targets `gepa.optimize_anything` within the exclusions below, not the full upstream surface. Upstream pin: `ce51b50cd196b539c25fae99ad0e0255c23004a4`.

## What "1-to-1 with Python" means

The acceptance gate runs selected, unmodified upstream pytests through a Python sidecar shim; [README.md](README.md#acceptance-gate) lists the exact targets. TS no-ops can fail sidecar-backed Python tests, but the shim also performs Python-side validation and compatibility work, so a green run is not full-suite or full-surface parity proof. As a consequence:

- **snake_case names are intentional**, not a style oversight. Preserve upstream-style names such as `optimize_anything`, `seed_candidate`, `reflection_lm`, and `make_reflective_dataset`; don't rename them to camelCase. Existing TS logging names include `oa_log`, `getLogContext`, and `setLogContext`, rather than Python's `log`, `get_log_context`, and `set_log_context`.
- Behavior should match upstream semantics, not "TypeScript-idiomatic" reinterpretations. When in doubt, the upstream Python is the spec.
- **Out of scope for v0.1**: checkpointing, evaluation cache, multimodal, refiner, reflection cost tracking, seedless mode. Adding any of these means either porting the upstream implementation or being explicit that it's a TS-only extension.

## Architecture

The genetic loop is a four-stage pipeline. Roughly: **engine** drives iterations, **candidate selector** picks a parent from the Pareto frontier, **proposer** mutates it via LLM reflection on a minibatch of evaluations, **adapter + evaluator** score the new candidate, **acceptance criterion** decides whether to admit it.

```
            ┌──────────────────────── engine.ts ────────────────────────┐
            │  GEPAEngine.run() — main loop, budget/stop, frontier      │
            │                                                           │
  state ───>│  candidate_selector ──> proposer ──> adapter ──> accept?  │──> state'
            │  (Pareto)              (reflective    (wraps              │
            │                         mutation)     evaluator)          │
            └───────────────────────────────────────────────────────────┘
```

Module map (all under `src/`):

- `index.ts` — public entry point. `optimize_anything(opts)` wires every component from `GEPAConfig` and runs the engine.
- `engine.ts` (`GEPAEngine`) — the genetic loop: budget tracking, iteration scheduling, valset evaluation, Pareto frontier maintenance, callback dispatch.
- `proposer.ts` (`ReflectiveMutationProposer`) — samples a minibatch, runs the parent through the adapter to collect traces, builds a reflective dataset, calls `reflection_lm` to propose new component texts.
- `adapter.ts` (`OptimizeAnythingAdapter`) — bridges the user's `evaluator` to the GEPA `GEPAAdapter` interface (`evaluate` + `make_reflective_dataset`). Extracts objective scores from `side_info` for reporting and reflection; the v0.1 selection frontier uses scalar per-instance scores.
- `evaluator_wrapper.ts` (`EvaluatorWrapper`) — normalizes user evaluator return shapes (`number` or `[number, side_info]`), captures stdout/stderr when `capture_stdio` is on, handles single-instance mode.
- `state.ts` — `GEPAState`: program tree, per-instance scores, frontier membership, metric-call count.
- `result.ts` — `GEPAResult` + `result_from_state` finalizer.
- `candidate_selector.ts` — `ParetoCandidateSelector` (the only built-in strategy accepted by `optimize_anything` in v0.1).
- `component_selector.ts` — `RoundRobinReflectionComponentSelector`, `AllReflectionComponentSelector`. Selects which component(s) to mutate this iteration.
- `eval_policy.ts` — `FullEvaluationPolicy` decides which valset items to score for a candidate.
- `acceptance.ts` — `StrictImprovementAcceptance`, `ImprovementOrEqualAcceptance`.
- `batch_sampler.ts` — `EpochShuffledBatchSampler` for reflection minibatch sampling.
- `data_loader.ts` — `ensure_loader` over user-provided datasets/valsets.
- `instruction_proposal.ts` + `reflection_prompt.ts` — prompt template for `reflection_lm`, plus `objective`/`background` injection.
- `log_context.ts` — `LogContext` / `oa_log` / `getLogContext` / `setLogContext` (AsyncLocalStorage-backed structured logging).
- `stoppers.ts` — `MaxMetricCallsStopper`, `CompositeStopper`.
- `utils.ts` — `SeededRandom` and small helpers.
- `types.ts` — shared API and configuration types. `OptimizeAnythingOpts` lives in `index.ts`; `GEPAResult` lives in `result.ts`.

`tests/unit/` mirrors `src/` filenames roughly 1-to-1 (vitest). Integration tests live in `tests/integration/`. The pytest-shim bridge lives in `tests/pytest-shim/` and `tests/run-pytest-shim.sh`.

## Bring your own LLM

Zero runtime deps and no built-in LLM adapter. Two functions are all you supply:

**1. `evaluator`** — given a candidate (and optionally an example + opt_state), return a score:

```ts
type Evaluator = {
  (candidate: string | Candidate): EvalResult | Promise<EvalResult>;
  (candidate: string | Candidate, ctx: { example?: unknown; opt_state?: EvaluatorOptState }):
    EvalResult | Promise<EvalResult>;
};
type EvalResult = number | [number, SideInfo];
```

Return a bare `number`, or `[score, side_info]`. The scalar score drives acceptance and the per-instance Pareto frontier. A top-level `side_info.scores` object and per-component `<param>_specific_info.scores` are collected for objective reporting and reflection; they do not drive candidate selection in v0.1.

**2. `reflection_lm`** (under `config.reflection.reflection_lm`) — a single async function:

```ts
type LanguageModel = (prompt: string | ChatMessage[]) => Promise<string>;
```

Wrap whatever you use (Anthropic, OpenAI, OpenRouter, local model) into that shape. There is no SDK dependency, no client object, no auth helper — just a function.

## Walkthrough: add a custom evaluator

This integration sketch optimizes a string seed against a tiny dataset, scoring each example and returning objective details for reporting and reflection. Supply your own async `callMyLLM(prompt: string, question: string)` and `callMyLLM_reflection(prompt: string | ChatMessage[])` functions returning strings; the sketch does not include a provider implementation. `ChatMessage` has the shape `{ role: string; content: string | unknown[] }`.

```ts
import { optimize_anything } from '@twaldin/gepa-ts';

type QA = { question: string; answer: string };

const dataset: QA[] = [
  { question: 'capital of France?', answer: 'Paris' },
  { question: '2 + 2?',             answer: '4' },
];

const result = await optimize_anything({
  seed_candidate: 'Answer the question concisely.',
  dataset,
  evaluator: async (candidate, ctx?: { example?: unknown }) => {
    const prompt = String(candidate);
    const { question, answer } = ctx?.example as QA;

    // call your LLM with `${prompt}\n\nQ: ${question}\nA:` ...
    const model_answer = await callMyLLM(prompt, question);

    const correct  = model_answer.trim().toLowerCase() === answer.toLowerCase();
    const concise  = model_answer.length < 80;
    const score    = (correct ? 1 : 0) * 0.8 + (concise ? 1 : 0) * 0.2;

    return [score, { scores: { correct: correct ? 1 : 0, concise: concise ? 1 : 0 } }];
  },
  config: {
    engine: { max_metric_calls: 50, seed: 0 },
    reflection: {
      reflection_lm: async (prompt) => callMyLLM_reflection(prompt),
      reflection_minibatch_size: 2,
    },
  },
});

console.log(result.best_candidate);
```

Notes on what to return from `evaluator`:

- A scalar score per example is sufficient for a diverse per-instance frontier: candidates can win on different validation examples. With only one validation instance, selection has only that one scalar dimension.
- `side_info.scores` (and/or `<component>_specific_info.scores`) adds objective detail for reports and reflection. v0.1 supports only `frontier_type: 'instance'`; objective, hybrid, and cartesian selection frontiers are not supported.
- `side_info` is also fed into the reflective dataset, so anything you put there (failure traces, raw model output, error messages) becomes context the `reflection_lm` can reflect on.

## The sidecar is test infrastructure, not the library

`src/sidecar/` (`server.ts`, `protocol.ts`, `main.ts`) is a unix-socket IPC bridge that lets selected, unmodified upstream Python pytests drive this TypeScript implementation as the system under test. The shim also supplies Python-side compatibility behavior; passing its selected tests is not proof that every path executes TS. The sidecar is **not** something a library user imports or runs — `optimize_anything` and the other public exports in `src/index.ts` are the user surface. When working in this repo, treat sidecar code as test harness; it does not belong in user-facing examples.

## Running the suites

- `bun run typecheck` — strict TS, must pass.
- `bun run test` — vitest, the full TS unit + integration suite.
- `bash tests/run-pytest-shim.sh` — the upstream pytest subset listed in [README.md](README.md#acceptance-gate), using the sidecar shim. A green vitest run does not replace this separate gate.

Build is `tsup` producing dual ESM + CJS with full `.d.ts`, targeting Node ≥20 and Bun ≥1. The current local build does not run directly under Deno: the emitted ESM imports bare `async_hooks`, which Deno rejects without Node-compatible resolution.
