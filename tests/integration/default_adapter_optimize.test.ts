import { describe, expect, it } from 'vitest';
import { DefaultAdapter, type DefaultDataInst } from '../../src/adapters/default_adapter/index.js';
import { optimize_anything } from '../../src/index.js';

describe('DefaultAdapter optimization path', () => {
  it('uses native adapter traces and reflection to discover an improved prompt', async () => {
    const data: DefaultDataInst[] = [
      { input: 'capital of France?', additional_context: {}, answer: 'Paris' },
      { input: '2 + 2?', additional_context: {}, answer: '4' },
    ];
    const adapter = new DefaultAdapter({
      model: async (messages) => {
        const system = messages[0]?.content ?? '';
        const question = messages[1]?.content ?? '';
        if (!system.includes('Answer with the exact expected answer')) {
          return 'I am unsure.';
        }
        return question.includes('France') ? 'Paris' : '4';
      },
    });

    const result = await optimize_anything({
      seed_candidate: { system_prompt: 'You are a helpful assistant.' },
      adapter,
      dataset: data,
      valset: data,
      config: {
        engine: { max_metric_calls: 8 },
        reflection: {
          reflection_minibatch_size: 2,
          reflection_lm: async () => '```\nAnswer with the exact expected answer.\n```',
        },
      },
    });

    expect(result.best_candidate).toEqual({
      system_prompt: 'Answer with the exact expected answer.',
    });
    expect(result.val_aggregate_scores[result.best_idx]).toBe(1);
    expect(result.candidates).toHaveLength(2);
    expect(result.parents[1]).toEqual([0]);
  });
});
