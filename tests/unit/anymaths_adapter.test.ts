import { describe, expect, it } from 'vitest';
import { AnyMathsAdapter } from '../../src/adapters/anymaths_adapter/index.js';

describe('AnyMathsAdapter', () => {
  it('evaluates structured math responses and builds reflective feedback', async () => {
    const adapter = new AnyMathsAdapter({
      model: 'ollama/qwen',
      completion_client: async ({ messages }) =>
        messages.map((_request, idx) => ({
          choices: [
            {
              message: {
                content:
                  idx === 0
                    ? JSON.stringify({ solution_pad: '2 + 2 = 4', final_answer: '4' })
                    : JSON.stringify({ solution_pad: '2 + 3 = 6', final_answer: '6' }),
              },
            },
          ],
        })),
    });

    const batch = [
      { input: '2 + 2?', additional_context: {}, answer: '4' },
      { input: '2 + 3?', additional_context: { hint: 'count carefully' }, answer: '5' },
    ];
    const result = await adapter.evaluate(batch, { instruction: 'Solve the problem.' }, true);

    expect(result.scores).toEqual([1, 0]);
    expect(result.outputs).toEqual([
      { full_assistant_response: "Assistant's Solution: 2 + 2 = 4\nFinal Answer: 4" },
      { full_assistant_response: "Assistant's Solution: 2 + 3 = 6\nFinal Answer: 6" },
    ]);
    expect(result.trajectories?.[0]?.data.input).toBe('2 + 2?');

    const reflective = adapter.make_reflective_dataset({ instruction: 'Solve the problem.' }, result, ['instruction']);
    expect(reflective.instruction).toEqual([
      {
        Inputs: '2 + 2?',
        'Generated Outputs': "Assistant's Solution: 2 + 2 = 4\nFinal Answer: 4",
        Feedback: 'The generated response is correct. The final answer is: 4.',
      },
      {
        Inputs: '2 + 3?',
        'Generated Outputs': "Assistant's Solution: 2 + 3 = 6\nFinal Answer: 6",
        Feedback:
          'The generated response is incorrect. The correct answer is: 5. Ensure that the correct answer is included in the response exactly as it is. Here is some additional context that might be helpful:\nhint: count carefully',
      },
    ]);
  });

  it('uses failure score for malformed responses and validates candidates', async () => {
    const adapter = new AnyMathsAdapter({
      model: 'openrouter/test',
      failure_score: -0.25,
      api_base: '',
      completion_client: async () => [{ choices: [{ message: { content: 'not json' } }] }],
    });

    await expect(adapter.evaluate([{ input: '1 + 1?', additional_context: {}, answer: '2' }], {}, false)).rejects.toThrow(
      'Candidate must contain at least one component text.',
    );

    const result = await adapter.evaluate(
      [{ input: '1 + 1?', additional_context: {}, answer: '2' }],
      { prompt: 'Solve.' },
      true,
    );

    expect(result.scores).toEqual([-0.25]);
    expect(result.outputs).toEqual([
      { full_assistant_response: 'Assistant failed to respond with the correct answer or format.' },
    ]);
    expect(result.trajectories?.[0]?.full_assistant_response).toBe(
      'Assistant failed to respond with the correct answer or format.',
    );
  });

  it('rejects missing callable client for zero-dependency string model execution', async () => {
    const adapter = new AnyMathsAdapter({ model: 'ollama/qwen' });

    await expect(
      adapter.evaluate([{ input: '1 + 1?', additional_context: {}, answer: '2' }], { prompt: 'Solve.' }),
    ).rejects.toThrow('String model execution is not bundled in the zero-dependency TypeScript adapter');
  });
});
