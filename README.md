# @twaldin/gepa-ts

A 1-1 TypeScript port of [gepa](https://github.com/gepa-ai/gepa) — Genetic-Pareto reflective text evolution for prompt and program optimization.

**Status: pre-v1, in active port.** Tracking upstream pin `ce51b50cd196b539c25fae99ad0e0255c23004a4`.

## Design constraints

- **1-1 with Python**: snake_case API, behavior-equivalent. The acceptance gate is the upstream pytest suite running unmodified against this implementation via a Python sidecar shim.
- **Bring your own LLM**: zero SDK dependencies. The reflection LM is a single function: `(prompt: string) => Promise<string>`. Wrap whatever you use (Anthropic, OpenAI, OpenRouter, local).
- **Zero runtime deps**: no transitive supply-chain footprint.
- **Cross-runtime**: Node ≥20, Bun ≥1, Deno. Dual ESM + CJS exports.

## v1 scope

`optimize_anything` and its full transitive closure: genetic loop, reflective mutation, candidate evaluation, Pareto frontier maintenance.

Anything outside that closure (adapter library, integrations) is post-v1.

## Install

Not yet published.
