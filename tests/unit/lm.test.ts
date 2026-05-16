import { describe, expect, it } from 'vitest';
import { LM, TrackingLM, ensure_tracking_lm, make_litellm_lm } from '../../src/lm.js';
import type { LanguageModel } from '../../src/types.js';

describe('TrackingLM', () => {
  it('wraps a language model and tracks estimated token usage', async () => {
    const calls: unknown[] = [];
    const lm = new TrackingLM(async (prompt) => {
      calls.push(prompt);
      return 'abcdefgh';
    });

    const result = await lm('abcdefgh');

    expect(result).toBe('abcdefgh');
    expect(calls).toEqual(['abcdefgh']);
    expect(lm.total_cost).toBe(0);
    expect(lm.total_tokens_in).toBe(2);
    expect(lm.total_tokens_out).toBe(2);
  });

  it('tracks chat-message prompts', async () => {
    const lm = new TrackingLM(async () => 'ok');

    await lm([{ role: 'user', content: 'hello' }]);

    expect(lm.total_tokens_in).toBeGreaterThan(1);
    expect(lm.total_tokens_out).toBe(1);
  });

  it('does not rewrap language models that already expose total_cost', () => {
    const lm: LanguageModel & { total_cost: number } = Object.assign(
      async () => 'ok',
      { total_cost: 1 },
    );

    expect(ensure_tracking_lm(lm)).toBe(lm);
  });
});

describe('LM', () => {
  it('calls an injected completion hook with upstream-shaped request fields and tracks usage', async () => {
    const lm = new LM('openai/test-model', {
      temperature: 0.2,
      max_tokens: 12,
      num_retries: 5,
      completion: (request) => {
        expect(request.model).toBe('openai/test-model');
        expect(request.messages).toEqual([{ role: 'user', content: 'hello' }]);
        expect(request.num_retries).toBe(5);
        expect(request.drop_params).toBe(true);
        expect(request.temperature).toBe(0.2);
        expect(request.max_tokens).toBe(12);
        return {
          choices: [{ message: { content: 'world' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 3, completion_tokens: 4 },
          cost: 0.25,
        };
      },
    });

    const result = await lm('hello');

    expect(result).toBe('world');
    expect(lm.total_cost).toBe(0.25);
    expect(lm.total_tokens_in).toBe(3);
    expect(lm.total_tokens_out).toBe(4);
    expect(String(lm)).toContain("LM(model='openai/test-model'");
  });

  it('supports chat messages and batch completion with injected hooks', async () => {
    const lm = make_litellm_lm('openai/batch-model', {
      completion: () => 'single',
      batch_completion: (request) => {
        expect(request.messages_list).toEqual([
          [{ role: 'user', content: 'a' }],
          [{ role: 'user', content: 'b' }],
        ]);
        expect(request.max_workers).toBe(2);
        return [
          { choices: [{ message: { content: 'A' } }], usage: { prompt_tokens: 1, completion_tokens: 2 }, cost: 0.1 },
          { choices: [{ message: { content: 'B' } }], usage: { prompt_tokens: 3, completion_tokens: 4 }, cost: 0.2 },
        ];
      },
    });

    await expect(lm([{ role: 'user', content: 'prompt' }])).resolves.toBe('single');
    await expect(lm.batch_complete([
      [{ role: 'user', content: 'a' }],
      [{ role: 'user', content: 'b' }],
    ], 2)).resolves.toEqual(['A', 'B']);
    expect(lm.total_cost).toBeCloseTo(0.3);
    expect(lm.total_tokens_in).toBeGreaterThan(4);
    expect(lm.total_tokens_out).toBeGreaterThan(6);
  });

  it('throws an actionable error when no completion hook is provided', async () => {
    const lm = new LM('openai/missing');

    await expect(lm('hello')).rejects.toThrow(/requires an injected completion hook/);
  });
});
