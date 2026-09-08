import type { Candidate, EvaluationBatch, GEPAAdapter } from '../../types.js';
import { LinearBlendScoring, type ScoringStrategy } from './scoring.js';

const TOP_ALTERNATIVES_IN_FEEDBACK = 3;

export type ConfidenceDataInst = {
  input: string;
  additional_context: Record<string, string>;
  answer: string;
};

export type ConfidenceTopAlternative = {
  token: string;
  probability: number;
  resolved_value?: string | null;
};

export type ConfidenceTrajectory = {
  data: ConfidenceDataInst;
  full_assistant_response: string;
  parsed_value: string | null;
  logprob_score: number | null;
  top_alternatives: ConfidenceTopAlternative[];
  is_correct: boolean;
  score: number;
  feedback: string;
};

export type ConfidenceRolloutOutput = {
  full_assistant_response: string;
  parsed_value: string | null;
  logprob_score: number | null;
};

export type ConfidenceChatMessage = {
  role: string;
  content: string;
};

export type ConfidenceModel = (messages: ConfidenceChatMessage[]) => unknown | Promise<unknown>;

export type ConfidenceLogprobExtraction = {
  joint_logprob: number;
  top_logprobs?: ConfidenceTopAlternative[];
};

export type ConfidenceLogprobExtractor = (
  response: unknown,
  args: { field_path: string; response_schema?: unknown },
) => ConfidenceLogprobExtraction[] | Promise<ConfidenceLogprobExtraction[]>;

export type ConfidenceAdapterConfig = {
  model: ConfidenceModel | string;
  field_path: string;
  response_format?: Record<string, unknown> | null;
  response_schema?: unknown;
  scoring_strategy?: ScoringStrategy | null;
  answer_field?: string | null;
  high_confidence_threshold?: number;
  low_confidence_threshold?: number;
  top_logprobs?: number;
  failure_score?: number;
  logprob_extractor?: ConfidenceLogprobExtractor | null;
};

function is_record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

export function _extract_answer_from_json(text: string, field_path: string): string | null {
  let obj: unknown;
  try {
    obj = JSON.parse(text);
  } catch {
    return null;
  }
  for (const key of field_path.split('.')) {
    if (!is_record(obj)) return null;
    obj = obj[key];
  }
  return obj === undefined || obj === null ? null : String(obj);
}

function _find_alternative_prob(alts: ConfidenceTopAlternative[], target: string): number | null {
  for (const alt of alts) {
    const value = alt.resolved_value ?? alt.token;
    if (value === target) return alt.probability;
  }
  return null;
}

function _format_alternatives(alts: ConfidenceTopAlternative[], exclude: string | null = null): string {
  const parts: string[] = [];
  for (const alt of alts.slice(0, TOP_ALTERNATIVES_IN_FEEDBACK)) {
    const value = alt.resolved_value ?? alt.token;
    if (value && value !== exclude) {
      parts.push(`'${value}' (${Math.round(alt.probability * 100)}%)`);
    }
  }
  return parts.join(', ');
}

export function _build_feedback({
  is_correct,
  expected,
  got,
  logprob_score,
  top_alternatives,
  additional_context,
  high_confidence_prob,
  low_confidence_prob,
}: {
  is_correct: boolean;
  expected: string;
  got: string | null;
  logprob_score: number | null;
  top_alternatives: ConfidenceTopAlternative[];
  additional_context: Record<string, string>;
  high_confidence_prob: number;
  low_confidence_prob: number;
}): string {
  const probability = logprob_score === null ? null : Math.exp(logprob_score);
  const got_str = got ?? '<parse error>';

  if (is_correct) {
    if (probability === null || probability >= high_confidence_prob) {
      return 'Correct.';
    }
    const alt_str = _format_alternatives(top_alternatives, expected);
    if (probability < low_confidence_prob) {
      let feedback =
        `Correct but uncertain (${Math.round(probability * 100)}% probability). ` +
        `Model answered '${expected}' but was nearly split with alternatives.`;
      if (alt_str) feedback += ` Top alternatives: ${alt_str}.`;
      return `${feedback} The model cannot reliably distinguish between these categories with the current prompt.`;
    }
    let feedback = `Correct (${Math.round(probability * 100)}% probability).`;
    if (alt_str) feedback += ` Close alternatives: ${alt_str}.`;
    return feedback;
  }

  const alt_str = _format_alternatives(top_alternatives, got_str);
  const correct_alt_prob = _find_alternative_prob(top_alternatives, expected);
  let feedback: string;
  if (probability !== null && probability >= high_confidence_prob) {
    feedback =
      `WRONG - model has ${Math.round(probability * 100)}% certainty on '${got_str}' ` +
      `but the correct answer is '${expected}'. ` +
      'The model has no doubt about its wrong answer; the prompt is actively misleading it for this type of input.';
    if (correct_alt_prob !== null) {
      feedback += ` The correct category '${expected}' only had ${(correct_alt_prob * 100).toFixed(1)}% probability.`;
    }
    if (alt_str) feedback += ` Alternatives: ${alt_str}.`;
    feedback += ` The prompt must add explicit rules to disambiguate '${got_str}' vs '${expected}'.`;
  } else if (probability !== null && probability >= low_confidence_prob) {
    feedback = `Wrong (${Math.round(probability * 100)}% probability). Expected '${expected}' but got '${got_str}'.`;
    if (alt_str) feedback += ` Alternatives: ${alt_str}.`;
    feedback += ' The prompt should better guide the model for this case.';
  } else {
    const prob_str = probability === null ? 'unknown confidence' : `${Math.round(probability * 100)}% probability`;
    feedback =
      `Wrong (${prob_str}). Expected '${expected}' but got '${got_str}'. ` +
      'The model was uncertain - better prompt guidance could fix this.';
    if (alt_str) feedback += ` Alternatives: ${alt_str}.`;
  }

  const ctx = Object.entries(additional_context).map(([key, value]) => `${key}: ${value}`).join('\n');
  return ctx ? `${feedback}\nAdditional context:\n${ctx}` : feedback;
}

function extract_text(response: unknown): string {
  if (typeof response === 'string') return response.trim();
  if (is_record(response)) {
    const choices = response['choices'];
    if (Array.isArray(choices) && choices.length > 0) {
      const first = choices[0];
      if (is_record(first)) {
        const message = first['message'];
        if (is_record(message) && typeof message['content'] === 'string') {
          return message['content'].trim();
        }
      }
    }
  }
  const with_choices = response as { choices?: Array<{ message?: { content?: unknown } }> };
  const content = with_choices.choices?.[0]?.message?.content;
  if (typeof content === 'string') return content.trim();
  throw new Error(`Cannot extract text from response of type ${typeof response}`);
}

export class ConfidenceAdapter implements GEPAAdapter<ConfidenceDataInst, ConfidenceTrajectory, ConfidenceRolloutOutput> {
  readonly model: ConfidenceModel | string;
  readonly field_path: string;
  readonly response_format: Record<string, unknown> | null;
  readonly response_schema: unknown;
  readonly scoring_strategy: ScoringStrategy;
  readonly answer_field: string;
  readonly high_confidence_threshold: number;
  readonly low_confidence_threshold: number;
  readonly top_logprobs: number;
  readonly failure_score: number;
  readonly logprob_extractor: ConfidenceLogprobExtractor | null;

  constructor({
    model,
    field_path,
    response_format = null,
    response_schema,
    scoring_strategy = null,
    answer_field = null,
    high_confidence_threshold = 0.99,
    low_confidence_threshold = 0.90,
    top_logprobs = 5,
    failure_score = 0.0,
    logprob_extractor = null,
  }: ConfidenceAdapterConfig) {
    if (typeof model === 'string' && response_format === null) {
      throw new Error('response_format is required when model is a string');
    }
    this.model = model;
    this.field_path = field_path;
    this.response_format = response_format;
    this.response_schema = response_schema;
    this.scoring_strategy = scoring_strategy ?? new LinearBlendScoring(high_confidence_threshold);
    this.answer_field = answer_field ?? field_path;
    this.high_confidence_threshold = high_confidence_threshold;
    this.low_confidence_threshold = low_confidence_threshold;
    this.top_logprobs = top_logprobs;
    this.failure_score = failure_score;
    this.logprob_extractor = logprob_extractor;
  }

  private async _call_model(messages: ConfidenceChatMessage[]): Promise<unknown> {
    if (typeof this.model === 'string') {
      throw new Error('String model execution is not bundled in the zero-dependency TypeScript adapter; pass a callable model.');
    }
    return this.model(messages);
  }

  private async _process_response(
    response: unknown,
    data: ConfidenceDataInst,
    capture_traces: boolean,
  ): Promise<[ConfidenceRolloutOutput, number, Record<string, number>, ConfidenceTrajectory | undefined]> {
    let response_text = '';
    let parsed_value: string | null = null;
    let logprob_score: number | null = null;
    let top_alternatives: ConfidenceTopAlternative[] = [];
    let is_correct = false;
    let score = this.failure_score;

    try {
      if (response instanceof Error) throw response;
      response_text = extract_text(response);
      parsed_value = _extract_answer_from_json(response_text, this.answer_field);
      if (this.logprob_extractor !== null) {
        const entries = await this.logprob_extractor(response, {
          field_path: this.field_path,
          response_schema: this.response_schema,
        });
        const first = entries[0];
        if (first !== undefined) {
          logprob_score = first.joint_logprob;
          top_alternatives = first.top_logprobs ?? [];
        }
      }
      is_correct = this._check_correctness(parsed_value, data.answer);
      score = this.scoring_strategy.score(is_correct, logprob_score);
    } catch {
      response_text = '';
      parsed_value = null;
      logprob_score = null;
      top_alternatives = [];
      is_correct = false;
      score = this.failure_score;
    }

    const feedback = _build_feedback({
      is_correct,
      expected: data.answer,
      got: parsed_value,
      logprob_score,
      top_alternatives,
      additional_context: data.additional_context ?? {},
      high_confidence_prob: this.high_confidence_threshold,
      low_confidence_prob: this.low_confidence_threshold,
    });

    const output = {
      full_assistant_response: response_text,
      parsed_value,
      logprob_score,
    };
    const probability = logprob_score === null ? 0.0 : Math.exp(logprob_score);
    const objective_scores = {
      accuracy: is_correct ? 1.0 : 0.0,
      probability,
    };

    const trajectory = capture_traces
      ? {
          data,
          full_assistant_response: response_text,
          parsed_value,
          logprob_score,
          top_alternatives,
          is_correct,
          score,
          feedback,
        }
      : undefined;

    return [output, score, objective_scores, trajectory];
  }

  async evaluate(
    batch: ConfidenceDataInst[],
    candidate: Candidate,
    capture_traces: boolean = false,
  ): Promise<EvaluationBatch<ConfidenceTrajectory, ConfidenceRolloutOutput>> {
    const system_content = Object.values(candidate)[0] ?? '';
    const outputs: ConfidenceRolloutOutput[] = [];
    const scores: number[] = [];
    const objective_scores: Array<Record<string, number>> = [];
    const trajectories: ConfidenceTrajectory[] | undefined = capture_traces ? [] : undefined;

    for (const data of batch) {
      const messages = [
        { role: 'system', content: system_content },
        { role: 'user', content: data.input },
      ];
      let response: unknown;
      try {
        response = await this._call_model(messages);
      } catch (error) {
        response = error instanceof Error ? error : new Error(String(error));
      }

      const [output, score, obj_scores, trajectory] = await this._process_response(response, data, capture_traces);
      outputs.push(output);
      scores.push(score);
      objective_scores.push(obj_scores);
      if (trajectories !== undefined && trajectory !== undefined) trajectories.push(trajectory);
    }

    return {
      outputs,
      scores,
      ...(trajectories !== undefined ? { trajectories } : {}),
      objective_scores,
      num_metric_calls: batch.length,
    };
  }

  make_reflective_dataset(
    _candidate: Candidate,
    eval_batch: EvaluationBatch<ConfidenceTrajectory, ConfidenceRolloutOutput>,
    components_to_update: string[],
  ): Record<string, Array<Record<string, unknown>>> {
    if (components_to_update.length !== 1) {
      throw new Error('ConfidenceAdapter expects exactly one component to update.');
    }
    const component = components_to_update[0]!;
    const trajectories = eval_batch.trajectories;
    if (trajectories === undefined) {
      throw new Error('Trajectories are required to build a reflective dataset.');
    }

    const items: Array<Record<string, unknown>> = [];
    for (const traj of trajectories) {
      let generated = traj.parsed_value ?? traj.full_assistant_response;
      if (traj.logprob_score !== null) {
        generated += ` (${Math.round(Math.exp(traj.logprob_score) * 100)}% probability)`;
      }
      items.push({
        Inputs: traj.data.input,
        'Generated Outputs': generated,
        Feedback: traj.feedback,
      });
    }
    if (items.length === 0) {
      throw new Error('No valid predictions found for any module.');
    }
    return { [component]: items };
  }

  static _check_correctness(parsed_value: string | null, expected: string): boolean {
    if (parsed_value === null) return false;
    return parsed_value.trim().toLowerCase() === expected.trim().toLowerCase();
  }

  _check_correctness(parsed_value: string | null, expected: string): boolean {
    return ConfidenceAdapter._check_correctness(parsed_value, expected);
  }
}
