import { describe, expect, it } from 'vitest';
import { optimize_anything } from '../../src/index.js';

describe('optimize_anything end-to-end (TS-only)', () => {
  it('runs the no-callbacks default path', async () => {
    const evaluator = (_candidate: string) => [0.5, {}] as [number, Record<string, unknown>];
    const reflection_lm = async (_prompt: string) => '```\ncandidate\n```';

    const result = await optimize_anything({
      seed_candidate: 'x',
      evaluator,
      config: {
        engine: { max_metric_calls: 3 },
        reflection: { reflection_lm },
      },
    });

    expect(result).not.toBeNull();
    expect(result.candidates.length).toBeGreaterThanOrEqual(1);
    expect(result.candidates[0]).toEqual({ current_candidate: 'x' });
  });
});
