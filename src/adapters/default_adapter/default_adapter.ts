import type { Candidate, EvaluationBatch, GEPAAdapter } from '../../types.js';

export type DefaultDataInst = {
  input: string;
  additional_context: Record<string, string>;
  answer: string;
};

export type EvaluationResult = {
  score: number;
  feedback: string;
  objective_scores?: Record<string, number> | null;
};

export type DefaultTrajectory = {
  data: DefaultDataInst;
  full_assistant_response: string;
  feedback: string;
};

export type DefaultRolloutOutput = {
  full_assistant_response: string;
};

export type DefaultChatMessage = {
  role: string;
  content: string;
};

export type DefaultChatCompletionCallable = (messages: DefaultChatMessage[]) => string | Promise<string>;
export type DefaultEvaluator = (data: DefaultDataInst, response: string) => EvaluationResult;

export class ContainsAnswerEvaluator {
  readonly failure_score: number;

  constructor(failure_score: number = 0.0) {
    this.failure_score = failure_score;
  }

  call(data: DefaultDataInst, response: string): EvaluationResult {
    const is_correct = response.includes(data.answer);
    if (is_correct) {
      return {
        score: 1.0,
        feedback: `The generated response is correct. The response include the correct answer '${data.answer}'`,
        objective_scores: null,
      };
    }

    const additional_context_str = Object.entries(data.additional_context ?? {})
      .map(([key, value]) => `${key}: ${value}`)
      .join('\n');
    let feedback =
      `The generated response is incorrect. The correct answer is '${data.answer}'. ` +
      'Ensure that the correct answer is included in the response exactly as it is.';
    if (additional_context_str) {
      feedback += ` Here is some additional context that might be helpful:\n${additional_context_str}`;
    }
    return { score: this.failure_score, feedback, objective_scores: null };
  }

  invoke(data: DefaultDataInst, response: string): EvaluationResult {
    return this.call(data, response);
  }
}

function normalize_evaluator(evaluator: DefaultEvaluator | ContainsAnswerEvaluator | null | undefined): DefaultEvaluator {
  if (evaluator === null || evaluator === undefined) {
    const contains = new ContainsAnswerEvaluator();
    return (data, response) => contains.call(data, response);
  }
  if (evaluator instanceof ContainsAnswerEvaluator) {
    return (data, response) => evaluator.call(data, response);
  }
  return evaluator;
}

export class DefaultAdapter implements GEPAAdapter<DefaultDataInst, DefaultTrajectory, DefaultRolloutOutput> {
  readonly model: DefaultChatCompletionCallable | string;
  readonly evaluator: DefaultEvaluator;
  readonly max_litellm_workers: number;
  readonly litellm_batch_completion_kwargs: Record<string, unknown>;

  constructor({
    model,
    evaluator = null,
    max_litellm_workers = 10,
    litellm_batch_completion_kwargs = null,
  }: {
    model: DefaultChatCompletionCallable | string;
    evaluator?: DefaultEvaluator | ContainsAnswerEvaluator | null;
    max_litellm_workers?: number;
    litellm_batch_completion_kwargs?: Record<string, unknown> | null;
  }) {
    this.model = model;
    this.evaluator = normalize_evaluator(evaluator);
    this.max_litellm_workers = max_litellm_workers;
    this.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs ?? {};
  }

  private async call_model(messages: DefaultChatMessage[]): Promise<string> {
    if (typeof this.model === 'string') {
      throw new Error('String model execution is not bundled in the zero-dependency TypeScript adapter; pass a callable model.');
    }
    return this.model(messages);
  }

  async evaluate(
    batch: DefaultDataInst[],
    candidate: Candidate,
    capture_traces: boolean = false,
  ): Promise<EvaluationBatch<DefaultTrajectory, DefaultRolloutOutput>> {
    const outputs: DefaultRolloutOutput[] = [];
    const scores: number[] = [];
    const objective_scores: Array<Record<string, number> | null> = [];
    const trajectories: DefaultTrajectory[] | undefined = capture_traces ? [] : undefined;
    const system_content = Object.values(candidate)[0] ?? '';

    for (const data of batch) {
      const assistant_response = await this.call_model([
        { role: 'system', content: system_content },
        { role: 'user', content: String(data.input) },
      ]);
      const eval_result = this.evaluator(data, assistant_response);
      outputs.push({ full_assistant_response: assistant_response });
      scores.push(eval_result.score);
      objective_scores.push(eval_result.objective_scores ?? null);
      if (trajectories !== undefined) {
        trajectories.push({
          data,
          full_assistant_response: assistant_response,
          feedback: eval_result.feedback,
        });
      }
    }

    let normalized_objective_scores: Array<Record<string, number>> | undefined;
    if (objective_scores.length > 0) {
      const all_none = objective_scores.every((entry) => entry === null);
      const all_not_none = objective_scores.every((entry) => entry !== null);
      if (!(all_none || all_not_none)) {
        throw new Error('Objective scores must either be all None or all not None.');
      }
      if (all_not_none) {
        normalized_objective_scores = objective_scores.map((entry) => entry ?? {});
      }
    }

    return {
      outputs,
      scores,
      ...(trajectories !== undefined ? { trajectories } : {}),
      ...(normalized_objective_scores !== undefined ? { objective_scores: normalized_objective_scores } : {}),
      num_metric_calls: batch.length,
    };
  }

  make_reflective_dataset(
    _candidate: Candidate,
    eval_batch: EvaluationBatch<DefaultTrajectory, DefaultRolloutOutput>,
    components_to_update: string[],
  ): Record<string, Array<Record<string, unknown>>> {
    if (components_to_update.length !== 1) {
      throw new Error('DefaultAdapter expects exactly one component to update.');
    }
    const component = components_to_update[0]!;
    const trajectories = eval_batch.trajectories;
    if (trajectories === undefined) {
      throw new Error('Trajectories are required to build a reflective dataset.');
    }

    const items = trajectories.map((traj) => ({
      Inputs: traj.data.input,
      'Generated Outputs': traj.full_assistant_response,
      Feedback: traj.feedback,
    }));
    if (items.length === 0) {
      throw new Error('No valid predictions found for any module.');
    }
    return { [component]: items };
  }
}
