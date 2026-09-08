import { describe, expect, test } from 'vitest';
import { ContainsAnswerEvaluator, DefaultAdapter, type DefaultDataInst } from '../../src/adapters/default_adapter/index.js';

const batch: DefaultDataInst[] = [
  { input: 'capital of France?', additional_context: { topic: 'geo' }, answer: 'Paris' },
  { input: '2 + 2?', additional_context: {}, answer: '4' },
];

describe('ContainsAnswerEvaluator', () => {
  test('scores by answer containment and includes context on failure', () => {
    const evaluator = new ContainsAnswerEvaluator(0.2);

    expect(evaluator.call(batch[0]!, 'The answer is Paris.').score).toBe(1);
    const failure = evaluator.call(batch[0]!, 'The answer is Lyon.');
    expect(failure.score).toBe(0.2);
    expect(failure.feedback).toContain("correct answer is 'Paris'");
    expect(failure.feedback).toContain('topic: geo');
  });
});

describe('DefaultAdapter', () => {
  test('calls callable model with system and user messages', async () => {
    const calls: unknown[] = [];
    const adapter = new DefaultAdapter({
      model: async (messages) => {
        calls.push(messages);
        return messages[1]?.content === 'capital of France?' ? 'Paris' : '4';
      },
    });

    const result = await adapter.evaluate(batch, { system_prompt: 'Answer exactly.' });

    expect(calls).toEqual([
      [{ role: 'system', content: 'Answer exactly.' }, { role: 'user', content: 'capital of France?' }],
      [{ role: 'system', content: 'Answer exactly.' }, { role: 'user', content: '2 + 2?' }],
    ]);
    expect(result.outputs).toEqual([
      { full_assistant_response: 'Paris' },
      { full_assistant_response: '4' },
    ]);
    expect(result.scores).toEqual([1, 1]);
    expect(result.objective_scores).toBeUndefined();
  });

  test('supports custom evaluator objective scores for Pareto tracking', async () => {
    const adapter = new DefaultAdapter({
      model: async () => 'Paris',
      evaluator: (data, response) => ({
        score: response.includes(data.answer) ? 1 : 0,
        feedback: response.includes(data.answer) ? 'Correct.' : 'Wrong.',
        objective_scores: { exact: response.includes(data.answer) ? 1 : 0, short: response.length < 20 ? 1 : 0 },
      }),
    });

    const result = await adapter.evaluate([batch[0]!], { system_prompt: 'Answer.' });

    expect(result.scores).toEqual([1]);
    expect(result.objective_scores).toEqual([{ exact: 1, short: 1 }]);
  });

  test('rejects mixed objective-score presence like upstream', async () => {
    const adapter = new DefaultAdapter({
      model: async (messages) => messages[1]?.content === 'capital of France?' ? 'Paris' : '4',
      evaluator: (data, response) => ({
        score: response.includes(data.answer) ? 1 : 0,
        feedback: 'ok',
        objective_scores: data.answer === 'Paris' ? { exact: 1 } : null,
      }),
    });

    await expect(adapter.evaluate(batch, { system_prompt: 'Answer.' })).rejects.toThrow('Objective scores must either be all None or all not None');
  });

  test('builds reflective dataset from captured trajectories', async () => {
    const adapter = new DefaultAdapter({
      model: async () => 'Lyon',
    });

    const eval_batch = await adapter.evaluate([batch[0]!], { system_prompt: 'Answer.' }, true);
    const dataset = adapter.make_reflective_dataset({ system_prompt: 'Answer.' }, eval_batch, ['system_prompt']);

    expect(dataset.system_prompt).toEqual([{
      Inputs: 'capital of France?',
      'Generated Outputs': 'Lyon',
      Feedback: eval_batch.trajectories?.[0]?.feedback,
    }]);
  });

  test('requires traces to build reflective dataset', async () => {
    const adapter = new DefaultAdapter({ model: async () => 'Paris' });
    const eval_batch = await adapter.evaluate([batch[0]!], { system_prompt: 'Answer.' });

    expect(() => adapter.make_reflective_dataset({ system_prompt: 'Answer.' }, eval_batch, ['system_prompt'])).toThrow('Trajectories are required');
  });
});
