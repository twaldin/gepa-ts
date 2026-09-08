import { describe, expect, it, vi } from 'vitest';
import {
  DSPyProgramProposalSignature,
  DspyAdapter,
} from '../../src/adapters/dspy_full_program_adapter/index.js';

describe('dspy_full_program_adapter', () => {
  it('returns list outputs and failure scores when the program cannot be built', async () => {
    const adapter = new DspyAdapter({
      task_lm: async () => 'unused',
      metric_fn: () => 1,
      reflection_lm: async () => 'unused',
      failure_score: -0.5,
      program_loader: () => ({
        program: null,
        feedback: 'Syntax Error in code: bad syntax',
      }),
    });

    const result = await adapter.evaluate([{ question: 'q1' }, { question: 'q2' }], { program: 'def foo(' });

    expect(result.outputs).toEqual([null, null]);
    expect(result.scores).toEqual([-0.5, -0.5]);
    expect(result.trajectories).toBe('Syntax Error in code: bad syntax');
    expect(Object.fromEntries(result.outputs.map((output, idx) => [idx, output]))).toEqual({ 0: null, 1: null });
  });

  it('uses evaluator hooks for valid programs', async () => {
    const adapter = new DspyAdapter({
      task_lm: async () => 'unused',
      metric_fn: () => 1,
      reflection_lm: async () => 'unused',
      program_loader: () => ({ program: { name: 'program' }, feedback: null }),
      evaluator: async ({ batch, program }) => ({
        outputs: batch.map((example) => ({ answer: `${String(example.question)}:${String(program.name)}` })),
        scores: batch.map(() => 1),
        trajectories: null,
      }),
    });

    const result = await adapter.evaluate([{ question: 'q1' }], { program: 'program = dspy.Predict("q -> a")' });

    expect(result.outputs).toEqual([{ answer: 'q1:program' }]);
    expect(result.scores).toEqual([1]);
  });

  it('renders and extracts full-program proposals', async () => {
    const dataset = [{ input: 'q1', output: 'a1', score: 0.5 }];
    const prompt = DSPyProgramProposalSignature.prompt_renderer({
      curr_program: 'program = dspy.Predict("q -> a")',
      dataset_with_feedback: dataset,
    });

    expect(prompt).toContain('program = dspy.Predict("q -> a")');
    expect(prompt).toContain('score: 0.5');
    expect(DSPyProgramProposalSignature.output_extractor('plan\n```\nprogram = improved\n```')).toEqual({
      new_program: 'program = improved',
    });
  });

  it('propose_new_texts calls reflection_lm through the proposal signature', async () => {
    const reflection_lm = vi.fn(async () => '```\nprogram = improved\n```');
    const adapter = new DspyAdapter({
      task_lm: async () => 'unused',
      metric_fn: () => 1,
      reflection_lm,
    });

    const result = await adapter.propose_new_texts(
      { program: 'program = seed' },
      { program: [{ input: 'q1', output: 'a1' }] },
      ['program'],
    );

    expect(result).toEqual({ program: 'program = improved' });
    expect(reflection_lm).toHaveBeenCalledOnce();
  });

  it('returns build feedback as reflective data', async () => {
    const adapter = new DspyAdapter({
      task_lm: async () => 'unused',
      metric_fn: () => 1,
      reflection_lm: async () => 'unused',
    });

    expect(
      adapter.make_reflective_dataset({ program: 'bad' }, { outputs: [], scores: [], trajectories: 'missing program' }, [
        'program',
      ]),
    ).toEqual({ program: { Feedback: 'missing program' } });
  });
});
