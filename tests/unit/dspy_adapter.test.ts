import { describe, expect, it, vi } from 'vitest';
import {
  DspyAdapter,
  InstructionProposalSignature,
  TOOL_MODULE_PREFIX,
  type DSPyProgram,
} from '../../src/adapters/dspy_adapter/index.js';

function make_program(): DSPyProgram {
  const make_predictor = () => ({
    signature: {
      instructions: 'old',
      with_instructions: (instructions: string) => ({ instructions }),
    },
  });
  return {
    tools: {
      search: { name: 'search', desc: 'old desc', args: { query: { description: 'old query' } } },
    },
    deepcopy: () => {
      const copied_predictor = make_predictor();
      return {
        tools: {
          search: { name: 'search', desc: 'old desc', args: { query: { description: 'old query' } } },
        },
        named_predictors: () => [['answer', copied_predictor]],
      };
    },
    named_predictors: () => [['answer', make_predictor()]],
  };
}

describe('dspy_adapter', () => {
  it('build_program updates predictor instructions and tool descriptions from candidates', () => {
    const adapter = new DspyAdapter({
      student_module: make_program(),
      metric_fn: () => 1,
      feedback_map: {},
      enable_tool_optimization: true,
    });

    const built = adapter.build_program({
      answer: 'new answer instruction',
      [`${TOOL_MODULE_PREFIX}:agent`]: JSON.stringify({
        answer: 'tool instruction',
        tools: { search: { desc: 'new desc', args: { query: { description: 'new query' } } } },
      }),
    });

    const predictors = Array.from(built.named_predictors?.() ?? []);
    expect(predictors[0]?.[1].signature?.instructions).toBe('tool instruction');
    expect(built.tools?.search?.desc).toBe('new desc');
    expect(built.tools?.search?.args?.query?.description).toBe('new query');
  });

  it('extracts scalar scores and subscores using upstream shapes', () => {
    expect(DspyAdapter._extract_score_and_subscores({ score: 0.7, subscores: { exact: 1, style: 0.4 } })).toEqual({
      score: 0.7,
      subscores: { exact: 1, style: 0.4 },
    });
    expect(DspyAdapter._extract_score_and_subscores(0.25)).toEqual({ score: 0.25, subscores: {} });
    expect(DspyAdapter._extract_score_and_subscores(null)).toEqual({ score: null, subscores: {} });
  });

  it('evaluate delegates DSPy execution to an injected evaluator and maps objective scores', async () => {
    const adapter = new DspyAdapter({
      student_module: make_program(),
      metric_fn: () => 1,
      feedback_map: {},
      evaluator: ({ batch }) => ({
        outputs: batch.map(() => ({ answer: 'a' })),
        scores: [0.8],
        trajectories: [],
      }),
    });

    const result = await adapter.evaluate([{ question: 'q' }], { answer: 'instruction' }, true);

    expect(result.outputs).toEqual([{ answer: 'a' }]);
    expect(result.scores).toEqual([0.8]);
    expect(result.num_metric_calls).toBe(1);
  });

  it('make_reflective_dataset formats traces and calls predictor feedback functions', () => {
    const feedback = vi.fn(() => ({ score: 0.5, feedback: 'mention citations' }));
    const adapter = new DspyAdapter({
      student_module: make_program(),
      metric_fn: () => 1,
      feedback_map: { answer: feedback },
    });

    const result = adapter.make_reflective_dataset(
      { answer: 'instruction' },
      {
        outputs: [{ final: 'x' }],
        scores: [0.5],
        trajectories: [
          {
            example: { question: 'q1' },
            prediction: { final: 'module out' },
            score: { score: 0.5 },
            trace: [
              {
                predictor: 'answer',
                inputs: { question: 'q1' },
                prediction: { answer: 'bad' },
              },
            ],
          },
        ],
      },
      ['answer'],
    );

    expect(result.answer).toEqual([
      {
        Inputs: { question: 'q1' },
        'Generated Outputs': { answer: 'bad' },
        Feedback: 'mention citations',
      },
    ]);
    expect(feedback).toHaveBeenCalledWith({
      predictor_output: { answer: 'bad' },
      predictor_inputs: { question: 'q1' },
      module_inputs: { question: 'q1' },
      module_outputs: { final: 'module out' },
      captured_trace: [
        {
          predictor: 'answer',
          inputs: { question: 'q1' },
          prediction: { answer: 'bad' },
        },
      ],
    });
  });

  it('keeps failed parses as feedback when enabled', () => {
    const adapter = new DspyAdapter({
      student_module: make_program(),
      metric_fn: () => 1,
      feedback_map: { answer: () => ({ score: 0, feedback: 'unused' }) },
      add_format_failure_as_feedback: true,
    });

    const result = adapter.make_reflective_dataset(
      { answer: 'instruction' },
      {
        outputs: [],
        scores: [],
        trajectories: [
          {
            trace: [
              {
                predictor: 'answer',
                inputs: { question: 'q1' },
                prediction: { failed_prediction: true, completion_text: 'raw text' },
              },
            ],
          },
        ],
      },
      ['answer'],
    );

    expect(result.answer?.[0]?.['Generated Outputs']).toContain('raw text');
    expect(result.answer?.[0]?.Feedback).toContain('failed to parse');
  });

  it('proposes regular instruction text through the instruction signature', async () => {
    const reflection_lm = vi.fn(async (prompt: string) => {
      expect(prompt).toContain('old instruction');
      expect(prompt).toContain('Feedback');
      return { text: 'new instruction' };
    });
    const adapter = new DspyAdapter({
      student_module: make_program(),
      metric_fn: () => 1,
      feedback_map: {},
      reflection_lm,
    });

    await expect(
      adapter.propose_new_texts({ answer: 'old instruction' }, { answer: [{ Feedback: 'too vague' }] }, ['answer']),
    ).resolves.toEqual({ answer: 'new instruction' });
  });

  it('normalizes stripped LM outputs and errors on missing text fields', () => {
    const adapter = new DspyAdapter({
      student_module: make_program(),
      metric_fn: () => 1,
      feedback_map: {},
      reflection_lm: () => [{ text: 'one' }, 'two'],
    });

    expect(adapter.stripped_lm_call('prompt')).toEqual(['one', 'two']);

    const bad_adapter = new DspyAdapter({
      student_module: make_program(),
      metric_fn: () => 1,
      feedback_map: {},
      reflection_lm: () => ({ reasoning: 'missing' }),
    });
    expect(() => bad_adapter.stripped_lm_call('prompt')).toThrow("Missing 'text' field");
  });

  it('renders instruction proposal prompts in a stable markdown shape', async () => {
    const prompt = InstructionProposalSignature.prompt_renderer({
      current_instruction_doc: 'answer carefully',
      dataset_with_feedback: [{ Inputs: { question: 'q' }, Feedback: 'wrong' }],
    });

    expect(prompt).toContain('## Current Instruction');
    expect(prompt).toContain('### question');
    expect(await InstructionProposalSignature.run({ lm: async () => 'better', input_dict: { current_instruction_doc: 'x', dataset_with_feedback: [] } })).toEqual({
      new_instruction: 'better',
    });
  });
});
