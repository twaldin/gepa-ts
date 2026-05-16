import { createHash } from "node:crypto";
import { type Candidate, type EvaluationBatch, type EvaluatorOptState, type LanguageModel, type RefinerConfig, type SideInfo } from "./types";

type ObjectiveScores = Record<string, number>;
type AttemptRecord = Record<string, unknown> & { iteration: number; score?: number };
type EvaluationTuple = [number, [number, Candidate, SideInfo], SideInfo];
type EvaluatorTriple = [number, unknown, SideInfo];

type WrappedEvaluator = {
  call(candidate: Candidate, example?: unknown, opt_state?: EvaluatorOptState): Promise<EvaluatorTriple>;
};

type AdapterEvaluatorFunction = (
  candidate: Candidate,
  example?: unknown,
  opt_state?: EvaluatorOptState,
) => EvaluatorTriple | Promise<EvaluatorTriple>;

type CacheMode = "off" | "memory" | "disk";

type BestExampleEval = { score: number; side_info: SideInfo };

type OptimizeAnythingAdapterParams = {
  evaluator: WrappedEvaluator | AdapterEvaluatorFunction;
  reflection_lm?: LanguageModel | null;
  reflection_prompt_template?: string | null;
  parallel?: boolean;
  max_workers?: number | null;
  refiner_config?: RefinerConfig | null;
  best_example_evals_k?: number;
  objective?: string | null;
  background?: string | null;
  cache_mode?: CacheMode;
  cache_dir?: string | null;
};

const REFINER_PROMPT_TEMPLATE = `You are refining a candidate to improve its performance.

## Instructions
{refiner_prompt}

## Current Candidate (JSON)
\`\`\`json
{candidate_to_improve}
\`\`\`

## Evaluation History
The following shows all evaluation attempts so far, including scores and feedback:
\`\`\`json
{evaluation_feedback}
\`\`\`

## Task
Analyze the evaluation history and propose an improved version of the candidate.
Return ONLY a valid JSON object with the improved parameters (no explanation, no markdown fences).
`;

function is_record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function wrap_evaluator(evaluator: WrappedEvaluator | AdapterEvaluatorFunction): WrappedEvaluator {
  return typeof evaluator === "function"
    ? {
        call: async (candidate: Candidate, example?: unknown, opt_state?: EvaluatorOptState): Promise<EvaluatorTriple> => evaluator(candidate, example, opt_state),
      }
    : evaluator;
}

function to_objective_scores(value: unknown): ObjectiveScores {
  if (!is_record(value)) {
    return {};
  }

  const ret: ObjectiveScores = {};
  for (const [key, raw_score] of Object.entries(value)) {
    if (typeof raw_score === "number") {
      ret[key] = raw_score;
    }
  }
  return ret;
}

function extract_objective_scores(side_info: SideInfo, candidate: Candidate): ObjectiveScores {
  const objective_score: ObjectiveScores = {};

  for (const [k, v] of Object.entries(to_objective_scores(side_info.scores))) {
    objective_score[k] = v;
  }

  for (const param_name of Object.keys(candidate)) {
    const specific_info = side_info[`${param_name}_specific_info`];
    if (!is_record(specific_info)) {
      continue;
    }

    const component_scores = to_objective_scores(specific_info.scores);
    for (const [k, v] of Object.entries(component_scores)) {
      objective_score[`${param_name}::${k}`] = v;
    }
  }

  return objective_score;
}

function strip_markdown_fence(text: string): string {
  const trimmed = text.trim();
  if (!trimmed.startsWith("```")) {
    return trimmed;
  }
  const lines = trimmed.split("\n");
  if (lines.length <= 1) {
    return trimmed.replace(/^```[a-zA-Z0-9_-]*/, "").replace(/```$/, "").trim();
  }
  const body = lines[lines.length - 1]?.trim() === "```" ? lines.slice(1, -1) : lines.slice(1);
  return body.join("\n").trim();
}

function parse_refinement(text: string): Candidate {
  const parsed = JSON.parse(strip_markdown_fence(text)) as unknown;
  if (!is_record(parsed)) {
    throw new Error("Expected JSON dict from refiner_lm");
  }
  const candidate: Candidate = {};
  for (const [key, value] of Object.entries(parsed)) {
    if (key !== "refiner_prompt") {
      candidate[key] = String(value);
    }
  }
  return candidate;
}

function format_refiner_prompt(refiner_prompt: string, candidate_to_improve: Candidate, attempts: AttemptRecord[]): string {
  return REFINER_PROMPT_TEMPLATE
    .replace("{refiner_prompt}", refiner_prompt)
    .replace("{candidate_to_improve}", JSON.stringify(candidate_to_improve, null, 2))
    .replace("{evaluation_feedback}", JSON.stringify(attempts, null, 2));
}

function resolve_refiner_lm(refiner_config: RefinerConfig | null): LanguageModel | null {
  if (refiner_config == null || typeof refiner_config.refiner_lm !== "function") {
    return null;
  }
  return refiner_config.refiner_lm;
}

function sorted_candidate_json(candidate: Candidate): string {
  return `[${Object.entries(candidate)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, value]) => `[${JSON.stringify(key)}, ${JSON.stringify(value)}]`)
    .join(", ")}]`;
}

function candidate_hash(candidate: Candidate): string {
  return createHash("sha256").update(sorted_candidate_json(candidate)).digest("hex").slice(0, 16);
}

function python_json_dumps(value: unknown): string {
  if (value === null) {
    return "null";
  }
  if (typeof value === "string") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new TypeError("Cannot JSON serialize non-finite number");
    }
    return Number.isInteger(value) ? String(value) : String(value);
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => python_json_dumps(item)).join(", ")}]`;
  }
  if (is_record(value)) {
    return `{${Object.entries(value)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, item]) => `${JSON.stringify(key)}: ${python_json_dumps(item)}`)
      .join(", ")}}`;
  }
  throw new TypeError(`Object of type ${typeof value} is not JSON serializable`);
}

function example_hash(example: unknown): string {
  if (example === undefined || example === null) {
    return "none";
  }
  try {
    return createHash("sha256").update(python_json_dumps(example)).digest("hex").slice(0, 16);
  } catch {
    return createHash("sha256").update(String(example)).digest("hex").slice(0, 16);
  }
}

function cache_key_parts(candidate: Candidate, example: unknown): [string, string] {
  return [candidate_hash(candidate), example_hash(example)];
}

function cache_key(candidate: Candidate, example: unknown): string {
  return cache_key_parts(candidate, example).join(":");
}

export class OptimizeAnythingAdapter {
  private readonly evaluator: WrappedEvaluator;
  readonly reflection_lm: LanguageModel | null;
  readonly reflection_prompt_template: string | null;
  readonly parallel: boolean;
  readonly max_workers: number | null;
  private readonly refiner_config: RefinerConfig | null;
  readonly best_example_evals_k: number;
  readonly objective: string | null;
  readonly background: string | null;
  readonly cache_mode: CacheMode;
  readonly cache_dir: string | null;
  private readonly best_evals_by_example = new Map<string, BestExampleEval[]>();
  private readonly eval_cache = new Map<string, EvaluatorTriple>();

  constructor(params: OptimizeAnythingAdapterParams);
  constructor(
    evaluator: WrappedEvaluator | AdapterEvaluatorFunction,
    reflection_lm?: LanguageModel | null,
    reflection_prompt_template?: string | null,
    parallel?: boolean,
    max_workers?: number | null,
    refiner_config?: RefinerConfig | null,
    best_example_evals_k?: number,
    objective?: string | null,
    background?: string | null,
    cache_mode?: CacheMode,
    cache_dir?: string | null,
  );
  constructor(
    params_or_evaluator: OptimizeAnythingAdapterParams | WrappedEvaluator | AdapterEvaluatorFunction,
    reflection_lm: LanguageModel | null = null,
    reflection_prompt_template: string | null = null,
    parallel = true,
    max_workers: number | null = null,
    refiner_config: RefinerConfig | null = null,
    best_example_evals_k = 1,
    objective: string | null = null,
    background: string | null = null,
    cache_mode: CacheMode = "memory",
    cache_dir: string | null = null,
  ) {
    const is_params_object = is_record(params_or_evaluator) && "evaluator" in params_or_evaluator;
    const params: OptimizeAnythingAdapterParams = is_params_object
      ? { cache_mode: "off", ...params_or_evaluator }
      : {
        evaluator: params_or_evaluator,
        reflection_lm,
        reflection_prompt_template,
        parallel,
        max_workers,
        refiner_config,
        best_example_evals_k,
        objective,
        background,
        cache_mode,
        cache_dir,
      };

    this.evaluator = wrap_evaluator(params.evaluator);
    this.reflection_lm = params.reflection_lm ?? null;
    this.reflection_prompt_template = params.reflection_prompt_template ?? null;
    this.parallel = params.parallel ?? true;
    this.max_workers = params.max_workers ?? null;
    this.refiner_config = params.refiner_config ?? null;
    this.best_example_evals_k = params.best_example_evals_k ?? 1;
    this.objective = params.objective ?? null;
    this.background = params.background ?? null;
    this.cache_mode = params.cache_mode ?? "memory";
    this.cache_dir = params.cache_dir ?? null;
  }

  _get_best_example_evals(example: unknown): BestExampleEval[] {
    return [...(this.best_evals_by_example.get(example_hash(example)) ?? [])].map((entry) => ({
      score: entry.score,
      side_info: { ...entry.side_info },
    }));
  }

  _update_best_example_evals(example: unknown, score: number, side_info: SideInfo): void {
    const key = example_hash(example);
    const current = this.best_evals_by_example.get(key) ?? [];
    current.push({ score, side_info: { ...side_info } });
    current.sort((a, b) => b.score - a.score);
    this.best_evals_by_example.set(key, current.slice(0, this.best_example_evals_k));
  }

  private _build_opt_state(example: unknown): EvaluatorOptState {
    return { best_example_evals: this._get_best_example_evals(example) };
  }

  _candidate_hash(candidate: Candidate): string {
    return candidate_hash(candidate);
  }

  _example_hash(example: unknown): string {
    return example_hash(example);
  }

  _cache_key(candidate: Candidate, example: unknown): [string, string] {
    return cache_key_parts(candidate, example);
  }

  _cache_filename(cache_key_value: [string, string]): string {
    return `${cache_key_value[0]}_${cache_key_value[1]}.pkl`;
  }

  _format_all_attempts_feedback(all_attempts: AttemptRecord[]): string {
    return JSON.stringify(all_attempts, null, 2);
  }

  private async _call_evaluator(candidate: Candidate, example: unknown, opt_state?: EvaluatorOptState): Promise<[number, unknown, SideInfo]> {
    const active_opt_state = opt_state ?? this._build_opt_state(example);
    if (this.cache_mode !== "off") {
      const key = cache_key(candidate, example);
      const cached = this.eval_cache.get(key);
      if (cached !== undefined) {
        return [cached[0], cached[1], { ...cached[2] }];
      }
      const result = await this.evaluator.call(candidate, example, active_opt_state);
      this.eval_cache.set(key, [result[0], result[1], { ...result[2] }]);
      return result;
    }
    return this.evaluator.call(candidate, example, active_opt_state);
  }

  async _evaluate_single_with_refinement(
    candidate: Candidate,
    example: unknown,
    opt_state?: EvaluatorOptState,
  ): Promise<{ score: number; output: [number, Candidate, SideInfo]; side_info: SideInfo; metric_calls: number }> {
    const refiner_lm = resolve_refiner_lm(this.refiner_config);
    const max_refinements = this.refiner_config?.max_refinements ?? 1;
    const refiner_prompt = candidate.refiner_prompt ?? "";
    const original_params: Candidate = {};
    for (const [key, value] of Object.entries(candidate)) {
      if (key !== "refiner_prompt") {
        original_params[key] = value;
      }
    }

    const [original_score, , original_side_info] = await this._call_evaluator(candidate, example, opt_state);
    this._update_best_example_evals(example, original_score, original_side_info);
    const attempts: AttemptRecord[] = [
      {
        iteration: 0,
        candidate: { ...original_params },
        score: original_score,
        side_info: { ...original_side_info },
      },
    ];

    let best_score = original_score;
    let best_candidate: Candidate = { ...candidate };
    let best_side_info: SideInfo = { ...original_side_info };
    let current_params: Candidate = { ...original_params };
    let metric_calls = 1;

    for (let iteration = 0; iteration < max_refinements; iteration += 1) {
      if (refiner_lm === null) {
        break;
      }
      const prompt = format_refiner_prompt(refiner_prompt, current_params, attempts);
      try {
        const raw_output = await refiner_lm(prompt);
        const parsed_refined = parse_refinement(String(raw_output));
        const refined_candidate: Candidate = {
          ...current_params,
          ...parsed_refined,
          refiner_prompt,
        };
        const [refined_score, , refined_side_info] = await this._call_evaluator(refined_candidate, example, opt_state);
        this._update_best_example_evals(example, refined_score, refined_side_info);
        metric_calls += 1;
        attempts.push({
          iteration: iteration + 1,
          candidate: { ...parsed_refined },
          score: refined_score,
          side_info: { ...refined_side_info },
        });
        if (refined_score >= best_score) {
          best_score = refined_score;
          best_candidate = refined_candidate;
          best_side_info = { ...refined_side_info };
          const next_params: Candidate = {};
          for (const [key, value] of Object.entries(refined_candidate)) {
            if (key !== "refiner_prompt") {
              next_params[key] = value;
            }
          }
          current_params = next_params;
        }
      } catch (error) {
        attempts.push({
          iteration: iteration + 1,
          error: String(error),
          score: -1000000000,
        });
        break;
      }
    }

    const aggregated_side_info: SideInfo = { ...best_side_info };
    const refiner_side_info: Record<string, unknown> = { Attempts: attempts };
    if (is_record(original_side_info.scores)) {
      const evaluated_attempts = attempts.filter((attempt) => is_record(attempt.side_info));
      let best_attempt: AttemptRecord | null = null;
      for (const attempt of evaluated_attempts) {
        if (best_attempt === null || (attempt.score ?? Number.NEGATIVE_INFINITY) > (best_attempt.score ?? Number.NEGATIVE_INFINITY)) {
          best_attempt = attempt;
        }
      }
      const best_attempt_side_info = is_record(best_attempt?.side_info) ? best_attempt.side_info : null;
      refiner_side_info.scores = to_objective_scores(best_attempt_side_info?.scores);
    }
    aggregated_side_info.refiner_prompt_specific_info = refiner_side_info;

    return {
      score: best_score,
      output: [best_score, best_candidate, aggregated_side_info],
      side_info: aggregated_side_info,
      metric_calls,
    };
  }

  async evaluate(
    batch: unknown[],
    candidate: Candidate,
    _capture_traces: boolean = false,
    opt_states?: Array<EvaluatorOptState | undefined>,
  ): Promise<EvaluationBatch> {
    const outputs: EvaluationTuple[1][] = [];
    const scores: number[] = [];
    const trajectories: SideInfo[] = [];
    const side_infos: SideInfo[] = [];
    const objective_scores: ObjectiveScores[] = [];
    let num_metric_calls = 0;

    for (const [idx, example] of batch.entries()) {
      let score: number;
      let output: [number, Candidate, SideInfo];
      let side_info: SideInfo;
      if (this.refiner_config !== null) {
        const refined = await this._evaluate_single_with_refinement(candidate, example, opt_states?.[idx]);
        score = refined.score;
        output = refined.output;
        side_info = refined.side_info;
        num_metric_calls += refined.metric_calls;
      } else {
        const [raw_score, , raw_side_info] = await this._call_evaluator(candidate, example, opt_states?.[idx]);
        score = raw_score;
        side_info = raw_side_info;
        output = [score, candidate, side_info];
        this._update_best_example_evals(example, score, side_info);
        num_metric_calls += 1;
      }
      outputs.push(output);
      scores.push(score);
      trajectories.push(side_info);
      side_infos.push(side_info);
      objective_scores.push(extract_objective_scores(side_info, candidate));
    }

    return {
      outputs,
      scores,
      trajectories,
      side_infos,
      objective_scores,
      num_metric_calls,
    };
  }

  make_reflective_dataset(
    _candidate: Candidate,
    eval_batch: EvaluationBatch<SideInfo, unknown>,
    components_to_update: string[],
  ): Record<string, Array<Record<string, unknown>>> {
    const side_infos = eval_batch.trajectories ?? [];
    const ret: Record<string, Array<Record<string, unknown>>> = {};

    for (const component_name of components_to_update) {
      ret[component_name] = [];

      for (const side_info of side_infos) {
        const record: Record<string, unknown> = {};

        for (const [k, v] of Object.entries(side_info)) {
          if (k === "scores") {
            record["Scores (Higher is Better)"] = v;
          } else if (k === `${component_name}_specific_info`) {
            if (is_record(v)) {
              for (const [specific_key, specific_value] of Object.entries(v)) {
                record[specific_key] = specific_value;
              }
            }
          } else if (k.endsWith("_specific_info")) {
            continue;
          } else {
            record[k] = v;
          }
        }

        ret[component_name].push(record);
      }
    }

    return ret;
  }
}
