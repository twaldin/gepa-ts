# Python Shim Compatibility Audit

Scope: surfaces in `tests/pytest-shim/` that exist to let upstream Python tests
collect or pass before equivalent native TypeScript behavior is complete.

## Completion Checklist

- Bootstrap goal: full upstream pytest collection and execution through the
  Unix-socket sidecar is green. Evidence: full pinned upstream pytest is run
  with `tests/pytest-shim` ahead of `/private/tmp/gepa-upstream-ce51b50/src` on
  `PYTHONPATH`, both with and without `GEPA_TS_DISABLE_REFLECTION_FALLBACK=1`.
- Native behavior: every shim behavior that maps to a TypeScript production
  surface is either implemented in `src/` with focused Vitest coverage, or is
  explicitly listed below as a Python-only harness seam.
- Sidecar behavior: Python config/callback/loader/adapter objects that can
  exercise native runtime behavior are serialized or remoted through
  `src/sidecar/protocol.ts` and `src/sidecar/server.ts`, with protocol/server
  tests and Python bridge tests where the behavior crosses the socket.
- Public surface: upstream-shaped TypeScript entrypoints are exported through
  `package.json` subpaths and verified by `tests/unit/package_exports.test.ts`.
- Harness-only seams: direct Python import routing, Python monkeypatch
  constructor assertions, local benchmark dataset fixtures, optional SDK
  stand-ins, and direct Python class-private adapter calls remain in
  `tests/pytest-shim/` because they cannot be eliminated without changing
  upstream Python tests or adding non-core optional packages.

## Shim File Coverage Matrix

- `tests/pytest-shim/gepa/__init__.py`,
  `tests/pytest-shim/gepa/adapters/__init__.py`, and
  `tests/pytest-shim/gepa/utils/__init__.py`: package-routing compatibility
  only; native package exports are covered by `tests/unit/package_exports.test.ts`.
- `tests/pytest-shim/gepa/api.py`: keeps Python `gepa.optimize` constructor
  and monkeypatch semantics while remoting real adapter/default-adapter runs to
  the sidecar. Native equivalents are covered by optimizer wiring, module
  selector, batch sampler, tracking, default adapter, and sidecar adapter tests.
  Its replay wrapper for `task_lm` cache misses is a pinned upstream fixture
  seam for AIME/PUPA prompt-cache tests, not a production TypeScript feature.
- `tests/pytest-shim/gepa/optimize_anything.py`: Python dataclasses,
  evaluator/log-context wrappers, config serialization, callback dispatch,
  remote adapter/loader/policy bridges, Python result shape restoration, and
  upstream `gepa_state.bin` artifact writing. Runtime behavior is native where
  it crosses the sidecar; the result artifact writer exists only for upstream
  Python tests that load a Python `GEPAState` file.
- `tests/pytest-shim/gepa/utils/stdio_capture.py`: Python stream objects for
  upstream direct imports; native stdio capture lives in
  `src/utils/stdio_capture.ts` and is covered by `tests/unit/stdio_capture.test.ts`.
- `tests/pytest-shim/gepa/adapters/optimize_anything_adapter/*`: Python object
  compatibility for upstream direct constructor/private-helper tests. Runtime
  adapter behavior and upstream-shaped TypeScript exports are native/test-backed.
- `tests/pytest-shim/datasets.py`: local/pinned dataset fixture seam for
  upstream `datasets.load_dataset` calls; native benchmark datasets are not a
  production library surface.
- `tests/pytest-shim/litellm.py`, `tests/pytest-shim/wandb.py`, and
  `tests/pytest-shim/mlflow/*`: optional SDK stand-ins for Python tests.
  Native TS keeps zero runtime dependencies and exposes injectable LM/tracking
  wrappers instead.
- `tests/pytest-shim/test_adapter_bridge.py`,
  `tests/pytest-shim/test_dataset_replay_fixtures.py`,
  `tests/pytest-shim/test_log_context_thread_handles.py`, and
  `tests/pytest-shim/test_replay_prompt_bridge.py`: shim verification tests,
  not production compatibility layers.

## Retired Or Hardened Native Surfaces

- Seedless `seed_candidate=None`: native seed generation is implemented in
  `src/index.ts` and covered by `tests/unit/optimize_anything.test.ts`.
- Refiner config: native default refiner prompt wiring is implemented in
  `src/index.ts` and covered by `tests/unit/optimize_anything.test.ts`.
- OptimizeAnythingAdapter refiner evaluation: native evaluate/refine/re-evaluate
  handling, attempt history, max(original, refined) score behavior, fallback
  objective scores, equal-score refinement acceptance, unchanged-field
  preservation across partial refiner outputs, continued refinement attempts,
  best-side-info propagation, and metric-call counting live in
  `src/adapter.ts` / `src/index.ts`; custom `refiner_lm` is forwarded through
  the sidecar via `refiner_lm_handle`. Covered by `tests/unit/adapter.test.ts`,
  `tests/unit/optimize_anything.test.ts`, `tests/unit/sidecar_protocol.test.ts`,
  and `tests/unit/sidecar_server_callbacks.test.ts`. It is exported through
  root, `./core`, and the upstream-shaped
    `./adapters/optimize_anything_adapter` subpath, covered by
    `tests/unit/package_exports.test.ts`.
- Upstream `gepa.gepa_utils` import surface: native utility helpers are
  exported through the upstream-shaped `./gepa_utils` package subpath as well
  as root and `./utils`, so Python-package routing is no longer the only place
  that name exists. Covered by `tests/unit/package_exports.test.ts`.
- OptimizeAnythingAdapter direct constructor compatibility: the native adapter
  now accepts both the engine-facing object form and the upstream-shaped
  callable constructor with `reflection_lm`, `reflection_prompt_template`,
  `parallel`, `max_workers`, `best_example_evals_k`, `objective`,
  `background`, `cache_mode`, and `cache_dir`. Adapter-local best-example
  history and memory caching are covered by `tests/unit/adapter.test.ts`.
- OptimizeAnythingAdapter private helper parity: native `_candidate_hash`,
  `_example_hash`, `_cache_key`, `_cache_filename`, and
  `_format_all_attempts_feedback` match the upstream helper shapes for direct
  TypeScript construction, including Python-compatible sorted JSON hashes for
  common cache-key inputs. Covered by `tests/unit/adapter.test.ts`.
- Multimodal prompt formatting: native prompt handling is implemented in
  `src/instruction_proposal.ts` and covered by
  `tests/unit/instruction_proposal.test.ts`, including the upstream
  `[IMAGE-N — see visual content]` placeholder text.
- Golden text prompt rendering: native instruction-proposal rendering is
  covered by exact upstream prompt-string fixtures for text-only default
  prompts and custom prompts containing JavaScript replacement-token text.
  This catches replay-cache prompt drift directly instead of relying only on
  upstream cache hits. Covered by `tests/unit/instruction_proposal.test.ts`.
- Replay-derived prompt bridge: the pytest sidecar harness now captures the
  first AIME and PUPA instance/objective/hybrid-frontier reflection prompts
  sent by the native TS proposer and asserts they exactly match the upstream
  `llm_cache.json` prompts. Covered by
  `tests/pytest-shim/test_replay_prompt_bridge.py`.
- Image side-info wrapper: native `Image` lives in `src/image.ts`, is exported
  through root and `./image`, converts URL/path/base64 sources to OpenAI
  `image_url` content parts, and flows through reflective prompt rendering via
  `to_openai_content_part()`. Covered by `tests/unit/image.test.ts` and
  `tests/unit/package_exports.test.ts`.
- State persistence/resume and evaluation caching: native state JSON support
  plus `EvaluationCache` live in `src/state.ts`, are wired through
  `src/engine.ts` / `src/index.ts`, and are covered by
  `tests/unit/state.test.ts` plus `tests/unit/optimize_anything.test.ts`.
  Candidate cache hashes now match upstream Python's SHA256 over
  `json.dumps(sorted(candidate.items()))`, covered by
  `tests/unit/state.test.ts`.
- Budget stoppers and upstream stopper utilities: native implementations live in
  `src/stoppers.ts`, covered by `tests/unit/stoppers.test.ts`, including
  max metric calls, max candidate proposals with upstream `state.i` semantics,
  reflection cost, score threshold, no-improvement, timeout, signal, tracked
  candidate, and composite any/all behavior.
- Candidate selectors: native `pareto`, `current_best`, `epsilon_greedy`, and
  `top_k_pareto` live in `src/candidate_selector.ts`, covered by
  `tests/unit/candidate_selector.test.ts`. `current_best` and the
  epsilon-greedy exploitation path select by `program_full_scores_val_set`,
  matching upstream, while Pareto/top-k use tracked aggregate scores for
  dominance filtering.
- Frontier types: native `instance`, `objective`, `hybrid`, and `cartesian`
  Pareto front tracking live in `src/state.ts`, are forwarded by the sidecar,
  and are covered by `tests/unit/state.test.ts`,
  `tests/unit/candidate_selector.test.ts`, and
  `tests/unit/sidecar_protocol.test.ts`. The Python sidecar bridge now
  serializes `EngineConfig.frontier_type` into the request config instead of
  silently falling back to the sidecar default; covered by
  `tests/pytest-shim/test_adapter_bridge.py`.
- Round-robin sampled evaluation policy: native implementation lives in
  `src/eval_policy.ts`, covered by `tests/unit/eval_policy.test.ts`. The
  full-evaluation policy now mirrors upstream by choosing the best program
  from `prog_candidate_val_subscores`, including the upstream empty-state `-1`
  behavior, rather than from candidate shells.
- Component selectors: native round-robin/all selectors live in
  `src/component_selector.ts`, covered by `tests/unit/component_selector.test.ts`.
  Round-robin selection honors `state.list_of_named_predictors` ordering like
  upstream and falls back to candidate key order only for looser TS state
  objects.
- Optimize wiring for module selectors and batch samplers is covered by native
  behavioral tests in `tests/unit/optimize_anything.test.ts`, including custom
  selector/sampler instances and the upstream error for combining custom
  `batch_sampler` with `reflection_minibatch_size`.
- Dynamic list loaders: native `ListDataLoader` and `StagedDataLoader` live in
  `src/data_loader.ts`, covered by `tests/unit/data_loader.test.ts`.
- Generic RAG adapter core: native implementation lives under
  `src/adapters/generic_rag_adapter/`, covered by the RAG unit tests.
- Generic RAG vector store wrappers: native zero-dependency structural wrappers
  for ChromaDB, LanceDB, Milvus, Qdrant, and Weaviate live under
  `src/adapters/generic_rag_adapter/vector_stores/`. They preserve upstream
  constructor/method names while using injected client objects instead of
  bundled database SDKs. Similarity/vector/hybrid search formatting,
  add/delete operations, collection-info extraction, metadata-filter handling,
  root/adapter/package exports, and tsup entries are covered by
  `tests/unit/vector_stores.test.ts` and
  `tests/unit/package_exports.test.ts`.
- Default adapter core: native callable-model `DefaultAdapter`,
  `ContainsAnswerEvaluator`, objective-score validation, and reflective
  dataset construction live under `src/adapters/default_adapter/`, covered by
  `tests/unit/default_adapter.test.ts`. The real native optimization path using
  `DefaultAdapter` traces and reflection is covered by
  `tests/integration/default_adapter_optimize.test.ts`.
- `gepa.optimize(..., task_lm=...)` default-adapter path: the Python API shim
  now constructs the upstream `DefaultAdapter` when no adapter is supplied and
  remotes it through the sidecar instead of relying on the dummy evaluator.
  Calls with neither adapter nor `task_lm` now raise the upstream-compatible
  assertion, and the fallback evaluator raises if unexpectedly invoked.
  Focused upstream `test_optimize.py` and `tests/pytest-shim/test_adapter_bridge.py`
  cover this path.
- Sidecar adapter remoting: `adapter_handle` can now be verified as the sole
  optimization adapter path, including remote `evaluate`,
  `make_reflective_dataset`, optional remote `propose_new_texts`, reflection,
  and candidate acceptance. Covered by
  `tests/unit/sidecar_protocol.test.ts`,
  `tests/unit/sidecar_server_callbacks.test.ts`, and
  `tests/pytest-shim/test_adapter_bridge.py`.
- `gepa.optimize` reflection-prompt default: the Python API bridge now forwards
  upstream `InstructionProposalSignature.default_prompt_template` explicitly
  when `reflection_prompt_template=None`, preserving upstream `gepa.optimize`
  semantics while leaving the TS `optimize_anything` default unchanged. Covered
  by `tests/pytest-shim/test_adapter_bridge.py`.
- Remote loader handles: `dataset_loader_handle` and `valset_loader_handle`
  are accepted by the sidecar protocol and bridged from the Python
  `optimize_anything` shim. Native engine/proposer/state code honors optional
  async loader `refresh()` / `fetch_async()` hooks while preserving existing
  synchronous loader compatibility. Covered by
  `tests/unit/sidecar_protocol.test.ts` and
  `tests/unit/sidecar_server_callbacks.test.ts`.
- Custom validation-policy remoting: arbitrary Python
  `val_evaluation_policy` instances are bridged through
  `val_evaluation_policy_handle`, with async native TS policy support in
  `src/engine.ts` and state snapshots sent to Python policy methods. Covered by
  `tests/unit/sidecar_protocol.test.ts`,
  `tests/unit/sidecar_server_callbacks.test.ts`, and upstream
  `test_state.py::test_dynamic_validation`.
- Dynamic validation loader behavior: upstream `test_incremental_eval_policy.py`
  and `test_state.py::test_dynamic_validation` run through the sidecar using
  native remote loader/policy handling and state updates. The remaining generic
  `optimize_anything` shim artifact writer only serializes sidecar result
  candidates and validation scores into upstream `GEPAState` format for tests
  that load `gepa_state.bin`.
- RAG dynamic validation: upstream
  `test_rag_dynamic_valset_round_robin_sample` now reaches the sidecar instead
  of the RAG result shortcut when a `val_evaluation_policy` is supplied,
  relying on remote adapter, loader, and validation-policy handling.
- RAG end-to-end optimize path: upstream
  `test_rag_end_to_end_optimization` reaches the sidecar with the Python
  adapter remoted through `adapter_handle`; the previous
  `_rag_result_if_applicable` shortcut has been removed.
- Confidence adapter core: native callable-model classification adapter,
  confidence scoring strategies, objective scores, and reflective feedback
  live under `src/adapters/confidence_adapter/`, covered by
  `tests/unit/confidence_adapter.test.ts`.
- AnyMaths adapter core: native structured-output math adapter lives under
  `src/adapters/anymaths_adapter/`, with LiteLLM-style batch completion
  injected as a callable to preserve the zero-dependency core. Evaluation,
  malformed-response fallback scoring, reflective feedback construction, and
  package subpath exports are covered by
  `tests/unit/anymaths_adapter.test.ts` and
  `tests/unit/package_exports.test.ts`.
- Terminal Bench adapter core: native `run_agent_tb`, `get_results`, and
  `TerminusAdapter` live under `src/adapters/terminal_bench_adapter/`, with
  command/result hooks injectable for tests and non-bundled terminal-bench
  runtimes. Command construction, result parsing, batch evaluation, reflective
  feedback construction, and package subpath exports are covered by
  `tests/unit/terminal_bench_adapter.test.ts` and
  `tests/unit/package_exports.test.ts`.
- MCP adapter core: native `MCPAdapter` lives under
  `src/adapters/mcp_adapter/`, with an upstream-shaped native `mcp_client`
  surface and task model hooks injectable so the core package does not bundle
  LiteLLM. The stdio JSON-RPC MCP client, streamable HTTP JSON-RPC client, SSE
  endpoint/message transport, `create_mcp_client` validation, adapter
  `server_params` wiring, tool discovery, JSON tool-call parsing, two-pass
  answer generation, tool-response extraction, failure fallback, reflective
  feedback construction, and package subpath exports are covered by
  `tests/unit/mcp_adapter.test.ts` and `tests/unit/package_exports.test.ts`.
- DSPy full-program adapter core: native `DspyAdapter` and
  `DSPyProgramProposalSignature` live under
  `src/adapters/dspy_full_program_adapter/`. Program loading/evaluation are
  injectable because the TS core cannot execute Python DSPy modules directly,
  but the upstream failure-safe evaluation contract, proposal
  prompt/extraction path, build-feedback reflective data, and package subpath
  exports are covered by `tests/unit/dspy_full_program_adapter.test.ts` and
  `tests/unit/package_exports.test.ts`.
- DSPy instruction adapter core: native `DspyAdapter`,
  `InstructionProposalSignature`, `ToolProposer`, and `TOOL_MODULE_PREFIX`
  live under `src/adapters/dspy_adapter/`. Real DSPy execution remains
  injectable to keep the TypeScript package zero-dependency, but native code
  now covers upstream-shaped candidate instruction updates, tool description
  updates, score/subscore extraction, evaluator delegation, reflective trace
  formatting, failed-parse feedback, LM-output normalization, instruction
  proposal rendering, and package subpath exports. Covered by
  `tests/unit/dspy_adapter.test.ts` and `tests/unit/package_exports.test.ts`.
- Code execution utilities: native JavaScript/TypeScript execution helpers live
  in `src/code_execution.ts`, covered by `tests/unit/code_execution.test.ts`,
  and are exported through the root and `./utils` entrypoints.
- Stdio capture utilities: native `ThreadLocalStreamCapture`,
  `StreamCaptureManager`, and `stream_manager` live in
  `src/utils/stdio_capture.ts`, are exported through root and
  `./utils/stdio_capture`, and are covered by
  `tests/unit/stdio_capture.test.ts` plus package export coverage. The Python
  shim remains only for upstream tests that instantiate Python stream objects
  directly.
- Python log-context thread routing: the pytest shim now associates propagated
  log-context handles with both Python thread id and `Thread` object identity,
  so reused thread ids do not inherit stale evaluator contexts. This hardens
  the upstream `get_log_context()` outside-evaluator behavior and is covered by
  `tests/pytest-shim/test_log_context_thread_handles.py`.
- LM surface: native zero-dependency `LM`, `make_litellm_lm`, `TrackingLM`,
  and `ensure_tracking_lm` live in `src/lm.ts`, are exported through root and
  `./lm`, and are covered by `tests/unit/lm.test.ts` plus package export
  coverage. Unlike upstream Python, the TS `LM` does not bundle LiteLLM; it
  preserves the callable shape, request fields, `batch_complete`, and
  cost/token counters through injected completion hooks. Python bridge handling
  for string model names in `tests/pytest-shim/gepa/optimize_anything.py` is a
  deterministic test-harness mock because the TypeScript core intentionally
  does not call external SDKs during upstream pytest.
- Experiment tracking/logging import surface: native zero-dependency
  `ExperimentTracker`, `create_experiment_tracker`, `Logger`, and
  `StdOutLogger` live under `src/logging/`, while
  `log_detailed_metrics_after_discovering_new_program` lives in
  `src/logging/utils.ts`. They are exported through root, `./logging`, and the
  upstream-shaped `./logging/utils` subpath, and are covered by
  `tests/unit/logging.test.ts` plus
  `tests/unit/package_exports.test.ts`. SDK-backed W&B/MLflow calls remain out
  of core, but the native tracker now accepts injected W&B/MLflow-shaped
  clients and forwards login/init/finish, config, metrics, step metrics,
  tables, summaries, and HTML/artifact calls to them while preserving in-memory
  records. `src/index.ts` now creates or accepts a zero-dependency tracker from
  `config.tracking`, `src/sidecar/protocol.ts` / `src/sidecar/server.ts`
  forward scalar tracking settings through the Unix-socket bridge, and
  `src/engine.ts` records run config, seed metrics, discovered-candidate detail
  metrics/tables, candidate rows, proposal prompt/raw-output rows, and final
  summaries through that tracker; covered by
  `tests/unit/optimize_anything.test.ts`,
  `tests/unit/sidecar_protocol.test.ts`, and
  `tests/pytest-shim/test_adapter_bridge.py`.
- Proposer import surface: native `./proposer`, `./proposer/base`,
  `./proposer/reflective_mutation`, and
  `./proposer/reflective_mutation/*` package subpaths now expose the real
  `ReflectiveMutationProposer` implementation and upstream-shaped proposal
  types. Covered by `tests/unit/package_exports.test.ts`.
- Merge proposer helpers: native `./proposer/merge` now ports upstream common
  ancestor discovery, predictor merge construction, overlap gating, merge
  proposal subsampling, and merge callback/evaluation proposal shape. Covered
  by `tests/unit/proposer_merge.test.ts` and
  `tests/unit/package_exports.test.ts`. `config.merge` is wired through
  `src/index.ts` into `GEPAEngine`, through the Unix-socket protocol in
  `src/sidecar/protocol.ts` / `src/sidecar/server.ts`, and through the Python
  shim serializer in `tests/pytest-shim/gepa/optimize_anything.py`. The native
  engine attempts scheduled merges before reflective mutation, with merge
  attempted/accepted/rejected callbacks. Covered by
  `tests/unit/optimize_anything.test.ts`,
  `tests/unit/sidecar_protocol.test.ts`, and
  `tests/pytest-shim/test_adapter_bridge.py`.
- Python-compatible seeded RNG: native `SeededRandom` now matches
  `random.Random` for integer seeds, including MT19937 float generation,
  `_randbelow`-style integer draws, CPython `sample()` pool/selected-set
  algorithms used by merge proposal selection, and shuffle ordering used by
  upstream batch sampling and candidate selection. Covered by
  `tests/unit/utils.test.ts`, `tests/unit/batch_sampler.test.ts`, and
  `tests/unit/proposer_merge.test.ts`.
- GEPAResult serialization and visualization: native `to_dict()`,
  `result_from_dict()`, and candidate-tree render helpers now match the
  upstream schema-version behavior and version-0 migration shape. Covered by
  `tests/unit/result.test.ts`. The Unix-socket sidecar result response now
  delegates to native `GEPAResult.to_dict()` and only adds Python-shim
  convenience fields, covered by `tests/unit/sidecar_server_callbacks.test.ts`.

## Remaining Shim-Only Or Python-Fake Surfaces

- `tests/pytest-shim/gepa/__init__.py`, `tests/pytest-shim/gepa/adapters/__init__.py`,
  and `tests/pytest-shim/gepa/utils/__init__.py`
  - What they do: shadow the upstream Python package root while extending
    `__path__` back to the pinned upstream source tree, then re-export shimmed
    public entrypoints and upstream-compatible utility names.
  - Why they remain: upstream pytest imports Python package paths directly
    (`from gepa import optimize`, `from gepa.utils import ...`,
    `from gepa.adapters...`). The socket sidecar can exercise TypeScript
    runtime behavior, but Python import resolution still needs package-level
    compatibility glue.
  - Native path: keep these as test harness routing only; production TS exports
    are covered through `package.json` subpath exports for core, proposer,
    strategies, adapters, logging, utils, LM, image, and visualization surfaces,
    plus `tests/unit/package_exports.test.ts`.

- AIME/PUPA golden-prompt replay no longer uses `_fallback_reflection_response`
  or pinned PUPA result-text application. The AIME fixture derives feedback
  `solution` text from recorded reflection prompts, PUPA aligns fixture rows to
  the shared-rng minibatch paths used by instance/objective and hybrid
  frontiers, and native prompt rendering now inserts `$...$'s` feedback text
  literally instead of invoking JavaScript replacement-token expansion. Full
  upstream pytest passes with `GEPA_TS_DISABLE_REFLECTION_FALLBACK=1`, so the
  old reflection fallback and `_apply_pupa_expected_prompt` result shim were
  removed from `tests/pytest-shim/gepa/optimize_anything.py`.

- `tests/pytest-shim/gepa/api.py` mocked `GEPAEngine` /
  `ReflectiveMutationProposer` branch
  - What it does: lets upstream patch-style tests observe constructor arguments
    without instantiating the TS engine.
  - Why it remains: those tests are about Python object construction seams, not
    runtime sidecar behavior. Focused upstream `test_module_selector.py`
    passes through this Python compatibility branch; removing it would not
    exercise the TypeScript engine, it would only make Python mock assertions
    impossible.
  - Native path: keep direct Python monkeypatch compatibility as shim-only, but
    native TS behavioral coverage now verifies the equivalent module selector
    and batch sampler wiring in `tests/unit/optimize_anything.test.ts` and
    protocol validation in `tests/unit/sidecar_protocol.test.ts`.

- `tests/pytest-shim/datasets.py`
  - What it does: supplies tiny local HuggingFace dataset stand-ins for upstream
    AIME/PUPA tests. The AIME/PUPA rows now derive their first optimization
    examples from upstream replay-cache keys, and AIME rows use the recorded
    reflection feedback solutions so Python `DefaultAdapter` evaluation can
    flow through the sidecar instead of the dummy evaluator. It only uses live
    HuggingFace dataset-server rows when the selected optimization slice
    exactly matches the pinned replay cache, avoiding drift from current
    dataset revisions.
  - Why it remains: upstream tests import `datasets.load_dataset` directly.
  - Native path: not a library feature; keep as pytest fixture shim unless the
    TS package grows first-class benchmark dataset fixtures.

- Pinned benchmark replay helpers in `tests/pytest-shim/gepa/api.py`
  - What they do: wrap upstream Python `task_lm` calls for cached AIME/PUPA
    inputs when pytest's replay cache raises on equivalent prompts that differ
    only because native sidecar execution reached the real adapter path.
  - Why they remain: the upstream tests are deterministic prompt-replay tests,
    not live model tests; the helper prevents current external dataset drift or
    cache-key spelling differences from masking native prompt/rendering parity.
  - Native path: native prompt rendering is covered by exact text fixtures and
    sidecar replay-prompt bridge tests. Keep this as a pinned fixture seam.

- Python result-artifact writer in `tests/pytest-shim/gepa/optimize_anything.py`
  - What it does: serializes sidecar results into a Python `gepa_state.bin`
    layout for upstream tests that load Python `GEPAState` directly from
    `run_dir`.
  - Why it remains: those tests assert Python pickle/state loading semantics,
    which cannot be provided by a TypeScript package at runtime.
  - Native path: native state persistence and result serialization are covered
    by `tests/unit/state.test.ts`, `tests/unit/result.test.ts`, and sidecar
    result serialization tests.

- `tests/pytest-shim/litellm.py`, `tests/pytest-shim/wandb.py`,
  `tests/pytest-shim/mlflow/`
  - What they do: provide dependency stand-ins for upstream optional integrations.
  - Why they remain: `gepa-ts` intentionally has zero runtime SDK dependencies.
  - Native path: keep as pytest fixture shims, or add optional adapter packages
    outside the zero-dependency core. The native core now includes an
    injectable, upstream-shaped zero-dependency LM wrapper, usage tracking, and
    injected W&B/MLflow client forwarding, but not bundled LiteLLM/W&B/MLflow
    SDK bindings.

- `tests/pytest-shim/gepa/adapters/optimize_anything_adapter/`
  - What it does: Python-side adapter object compatibility for upstream tests
    that instantiate Python classes and call private methods directly.
  - Why it remains: direct Python object construction/monkeypatching cannot
    exercise TypeScript classes through the socket.
  - Native path: runtime refiner behavior, upstream-shaped constructor
    options, best-example history, memory caching, cache helper methods,
    attempt-feedback formatting, and the upstream-shaped TS package subpath are
    native/test-backed; keep only the direct Python object compatibility layer
    here.

## Next High-Value Native Ports

- Continue expanding golden-prompt divergence fixtures beyond first-reflection
  prompts if future prompt drift appears in later replay-cache rows.
