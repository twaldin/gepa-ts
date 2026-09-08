import type { Candidate, EvaluationBatch, GEPAAdapter } from '../../types.js';

export type AnyMathsDataInst = {
  input: string;
  additional_context: Record<string, string>;
  answer: string;
};

export type AnyMathsTrajectory = {
  data: AnyMathsDataInst;
  full_assistant_response: string;
};

export type AnyMathsRolloutOutput = {
  full_assistant_response: string;
};

export type AnyMathsStructuredOutput = {
  final_answer: string;
  solution_pad: string;
};

export type AnyMathsChatMessage = {
  role: 'system' | 'user' | string;
  content: string;
};

export type AnyMathsCompletionResponse = {
  choices?: Array<{
    message?: {
      content?: string | null;
    } | null;
  }>;
};

export type AnyMathsCompletionRequest = {
  model: string;
  messages: AnyMathsChatMessage[][];
  api_base?: string | null;
  max_workers: number;
  format: Record<string, unknown>;
  response_format: {
    type: 'json_object';
    response_schema: Record<string, unknown>;
    enforce_validation: true;
  };
};

export type AnyMathsCompletionClient = (
  request: AnyMathsCompletionRequest,
) => AnyMathsCompletionResponse[] | Promise<AnyMathsCompletionResponse[]>;

const ANYMATHS_STRUCTURED_OUTPUT_SCHEMA: Record<string, unknown> = {
  type: 'object',
  required: ['final_answer', 'solution_pad'],
  properties: {
    final_answer: {
      type: 'string',
      description: 'The final answer to the mathematical problem (i.e., no units, no other text)',
    },
    solution_pad: {
      type: 'string',
      description: 'The solution pad containing the step-by-step solution to the problem.',
    },
  },
};

function parse_structured_output(content: string | null | undefined): AnyMathsStructuredOutput | null {
  if (typeof content !== 'string') return null;
  try {
    const parsed: unknown = JSON.parse(content.trim());
    if (typeof parsed !== 'object' || parsed === null) return null;
    const candidate = parsed as Record<string, unknown>;
    if (typeof candidate.final_answer !== 'string' || typeof candidate.solution_pad !== 'string') return null;
    return {
      final_answer: candidate.final_answer,
      solution_pad: candidate.solution_pad,
    };
  } catch {
    return null;
  }
}

export class AnyMathsAdapter implements GEPAAdapter<AnyMathsDataInst, AnyMathsTrajectory, AnyMathsRolloutOutput> {
  readonly model: string;
  readonly failure_score: number;
  readonly api_base: string | null;
  readonly max_litellm_workers: number;
  readonly completion_client: AnyMathsCompletionClient | null;

  constructor({
    model,
    failure_score = 0.0,
    api_base = 'http://localhost:11434',
    max_litellm_workers = 10,
    completion_client = null,
  }: {
    model: string;
    failure_score?: number;
    api_base?: string | null;
    max_litellm_workers?: number;
    completion_client?: AnyMathsCompletionClient | null;
  }) {
    this.model = model;
    this.failure_score = failure_score;
    if (this.model.startsWith('ollama')) {
      if (api_base === null || api_base === '') {
        throw new Error('API base URL must be provided when using Ollama.');
      }
    }
    this.api_base = api_base === '' ? null : api_base;
    this.max_litellm_workers = max_litellm_workers;
    this.completion_client = completion_client;
  }

  private async batch_completion(messages: AnyMathsChatMessage[][]): Promise<AnyMathsCompletionResponse[]> {
    if (this.completion_client === null) {
      throw new Error(
        'String model execution is not bundled in the zero-dependency TypeScript adapter; pass completion_client.',
      );
    }
    return this.completion_client({
      model: this.model,
      messages,
      api_base: this.api_base,
      max_workers: this.max_litellm_workers,
      format: ANYMATHS_STRUCTURED_OUTPUT_SCHEMA,
      response_format: {
        type: 'json_object',
        response_schema: ANYMATHS_STRUCTURED_OUTPUT_SCHEMA,
        enforce_validation: true,
      },
    });
  }

  async evaluate(
    batch: AnyMathsDataInst[],
    candidate: Candidate,
    capture_traces: boolean = false,
  ): Promise<EvaluationBatch<AnyMathsTrajectory, AnyMathsRolloutOutput>> {
    const outputs: AnyMathsRolloutOutput[] = [];
    const scores: number[] = [];
    const trajectories: AnyMathsTrajectory[] | undefined = capture_traces ? [] : undefined;

    const system_content = Object.values(candidate)[0];
    if (system_content === undefined) {
      throw new Error('Candidate must contain at least one component text.');
    }

    const litellm_requests = batch.map((data) => [
      { role: 'system', content: system_content },
      { role: 'user', content: data.input },
    ]);
    const responses = await this.batch_completion(litellm_requests);

    for (const [idx, data] of batch.entries()) {
      const content = responses[idx]?.choices?.[0]?.message?.content;
      const assistant_response = parse_structured_output(content);

      let output: AnyMathsRolloutOutput;
      let score: number;
      if (assistant_response !== null) {
        const structured_assistant_response =
          `Assistant's Solution: ${assistant_response.solution_pad}\n` +
          `Final Answer: ${assistant_response.final_answer}`;
        output = { full_assistant_response: structured_assistant_response };
        score = assistant_response.final_answer.includes(data.answer) ? 1.0 : this.failure_score;
      } else {
        output = { full_assistant_response: 'Assistant failed to respond with the correct answer or format.' };
        score = this.failure_score;
      }

      outputs.push(output);
      scores.push(score);
      if (trajectories !== undefined) {
        trajectories.push({
          data,
          full_assistant_response: output.full_assistant_response,
        });
      }
    }

    return {
      outputs,
      scores,
      ...(trajectories !== undefined ? { trajectories } : {}),
      num_metric_calls: batch.length,
    };
  }

  make_reflective_dataset(
    _candidate: Candidate,
    eval_batch: EvaluationBatch<AnyMathsTrajectory, AnyMathsRolloutOutput>,
    components_to_update: string[],
  ): Record<string, Array<Record<string, unknown>>> {
    if (components_to_update.length !== 1) {
      throw new Error('AnyMathsAdapter expects exactly one component to update.');
    }
    const comp = components_to_update[0]!;
    const trajectories = eval_batch.trajectories;
    if (trajectories === undefined) {
      throw new Error('Trajectories are required to build a reflective dataset.');
    }

    const items: Array<Record<string, unknown>> = [];
    for (const [idx, traj] of trajectories.entries()) {
      const score = eval_batch.scores[idx] ?? this.failure_score;
      const data = traj.data;
      let feedback: string;
      if (score > 0.0) {
        feedback = `The generated response is correct. The final answer is: ${data.answer}.`;
      } else {
        const additional_context_str = Object.entries(data.additional_context ?? {})
          .map(([key, value]) => `${key}: ${value}`)
          .join('\n');
        feedback =
          `The generated response is incorrect. The correct answer is: ${data.answer}. ` +
          'Ensure that the correct answer is included in the response exactly as it is.';
        if (additional_context_str) {
          feedback += ` Here is some additional context that might be helpful:\n${additional_context_str}`;
        }
      }

      items.push({
        Inputs: data.input,
        'Generated Outputs': traj.full_assistant_response,
        Feedback: feedback,
      });
    }

    if (items.length === 0) {
      throw new Error('No valid predictions found for any module.');
    }
    return { [comp]: items };
  }
}
