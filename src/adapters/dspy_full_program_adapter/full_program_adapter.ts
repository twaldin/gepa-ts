import type { Candidate, EvaluationBatch } from '../../types.js';
import { DSPyProgramProposalSignature } from './dspy_program_proposal_signature.js';

export type DSPyFullProgramExample = Record<string, unknown>;
export type DSPyFullProgramPrediction = Record<string, unknown> | null;
export type DSPyFullProgramTraceData = Record<string, unknown>;

export type DSPyProgramLoadResult = {
  program: Record<string, unknown> | null;
  feedback: string | null;
};

export type DSPyFullProgramEvaluationBatch = Omit<
  EvaluationBatch<DSPyFullProgramTraceData, DSPyFullProgramPrediction>,
  'trajectories'
> & {
  trajectories?: DSPyFullProgramTraceData[] | string | null;
};

export type DSPyProgramLoader = (candidate_src: string, context: Record<string, unknown>) => DSPyProgramLoadResult;
export type DSPyFullProgramEvaluator = (args: {
  batch: DSPyFullProgramExample[];
  candidate: Candidate;
  program: Record<string, unknown>;
  capture_traces: boolean;
}) => EvaluationBatch<DSPyFullProgramTraceData, DSPyFullProgramPrediction> | Promise<EvaluationBatch<DSPyFullProgramTraceData, DSPyFullProgramPrediction>>;

export type DSPyFullProgramAdapterConfig = {
  task_lm: unknown;
  metric_fn: (example: DSPyFullProgramExample, prediction: DSPyFullProgramPrediction, trace?: unknown) => number;
  reflection_lm: (prompt: string) => string | Promise<string>;
  failure_score?: number;
  num_threads?: number | null;
  add_format_failure_as_feedback?: boolean;
  program_loader?: DSPyProgramLoader | null;
  evaluator?: DSPyFullProgramEvaluator | null;
};

export class DspyAdapter {
  readonly task_lm: unknown;
  readonly metric_fn: (example: DSPyFullProgramExample, prediction: DSPyFullProgramPrediction, trace?: unknown) => number;
  readonly reflection_lm: (prompt: string) => string | Promise<string>;
  readonly failure_score: number;
  readonly num_threads: number | null;
  readonly add_format_failure_as_feedback: boolean;
  private readonly program_loader: DSPyProgramLoader | null;
  private readonly evaluator: DSPyFullProgramEvaluator | null;

  constructor({
    task_lm,
    metric_fn,
    reflection_lm,
    failure_score = 0.0,
    num_threads = null,
    add_format_failure_as_feedback = false,
    program_loader = null,
    evaluator = null,
  }: DSPyFullProgramAdapterConfig) {
    if (reflection_lm === null || reflection_lm === undefined) {
      throw new Error('DspyAdapter for full-program evolution requires a reflection_lm to be provided');
    }
    this.task_lm = task_lm;
    this.metric_fn = metric_fn;
    this.reflection_lm = reflection_lm;
    this.failure_score = failure_score;
    this.num_threads = num_threads;
    this.add_format_failure_as_feedback = add_format_failure_as_feedback;
    this.program_loader = program_loader;
    this.evaluator = evaluator;
  }

  build_program(candidate: Candidate): DSPyProgramLoadResult {
    const candidate_src = candidate.program;
    if (candidate_src === undefined) {
      return { program: null, feedback: 'Candidate did not include a `program` component.' };
    }
    if (this.program_loader !== null) {
      return this.program_loader(candidate_src, {});
    }
    if (!candidate_src.includes('program')) {
      return {
        program: null,
        feedback:
          'Your code did not define a `program` object. Please define a `program` object which is an instance of `dspy.Module`, either directly by dspy.Predict or dspy.ChainOfThought, or by instantiating a class that inherits from `dspy.Module`.',
      };
    }
    return {
      program: null,
      feedback: 'DSPy program execution is not bundled in the zero-dependency TypeScript adapter; pass program_loader and evaluator hooks.',
    };
  }

  async evaluate(
    batch: DSPyFullProgramExample[],
    candidate: Candidate,
    capture_traces: boolean = false,
  ): Promise<DSPyFullProgramEvaluationBatch> {
    const { program, feedback } = this.build_program(candidate);
    if (program === null) {
      return {
        outputs: batch.map(() => null),
        scores: batch.map(() => this.failure_score),
        trajectories: feedback,
        num_metric_calls: batch.length,
      };
    }
    if (this.evaluator === null) {
      return {
        outputs: batch.map(() => null),
        scores: batch.map(() => this.failure_score),
        trajectories: 'DSPy evaluation hook was not provided.',
        num_metric_calls: batch.length,
      };
    }
    const result = await this.evaluator({ batch, candidate, program, capture_traces });
    return { ...result, num_metric_calls: result.num_metric_calls ?? batch.length };
  }

  make_reflective_dataset(
    _candidate: Candidate,
    eval_batch: EvaluationBatch<DSPyFullProgramTraceData, DSPyFullProgramPrediction> & {
      trajectories?: DSPyFullProgramTraceData[] | string | null;
    },
    components_to_update: string[],
  ): Record<string, unknown> {
    if (new Set(components_to_update).size !== 1 || components_to_update[0] !== 'program') {
      throw new Error(`set(components_to_update) = ${JSON.stringify([...new Set(components_to_update)])}`);
    }
    if (typeof eval_batch.trajectories === 'string') {
      return { program: { Feedback: eval_batch.trajectories } };
    }
    const items: Array<Record<string, unknown>> = [];
    for (const [idx, trajectory] of (eval_batch.trajectories ?? []).entries()) {
      const example = record_at(trajectory, 'example');
      const prediction = record_at(trajectory, 'prediction');
      const trace = Array.isArray(trajectory.trace) ? trajectory.trace : [];
      const example_data: Record<string, unknown> = {
        'Program Inputs': record_at(example, 'inputs'),
        'Program Outputs': prediction,
        'Program Trace': trace,
      };
      const score = record_at(trajectory, 'score');
      if (typeof score.feedback === 'string') {
        example_data.Feedback = score.feedback;
      }
      if (idx < eval_batch.outputs.length) {
        items.push(example_data);
      }
    }
    if (items.length === 0) {
      throw new Error('No valid predictions found for program.');
    }
    return { program: items };
  }

  async propose_new_texts(
    candidate: Candidate,
    reflective_dataset: Record<string, Array<Record<string, unknown>>>,
    components_to_update: string[],
  ): Promise<Candidate> {
    const new_texts: Candidate = {};
    for (const name of components_to_update) {
      const base_instruction = candidate[name];
      const dataset_with_feedback = reflective_dataset[name];
      if (base_instruction === undefined || dataset_with_feedback === undefined) {
        continue;
      }
      const result = await DSPyProgramProposalSignature.run({
        lm: this.reflection_lm,
        input_dict: { curr_program: base_instruction, dataset_with_feedback },
      });
      new_texts[name] = result.new_program;
    }
    return new_texts;
  }
}

function record_at(value: unknown, key: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null) return {};
  const record = value as Record<string, unknown>;
  const child = record[key];
  if (typeof child === 'object' && child !== null) {
    return child as Record<string, unknown>;
  }
  return {};
}
