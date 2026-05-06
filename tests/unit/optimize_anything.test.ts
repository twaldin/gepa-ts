import { describe, it, expect } from 'vitest';
import { optimize_anything } from '../../src/index.js';

describe('optimize_anything reflection_lm validation', () => {
  it('throws when reflection_lm is missing', async () => {
    await expect(optimize_anything({
      seed_candidate: 'x',
      evaluator: () => [0.5, {}],
      config: { engine: { max_metric_calls: 3 }, reflection: {} },
    })).rejects.toThrow(/reflection_lm/);
  });

  it('throws when reflection_lm is not a function', async () => {
    await expect(optimize_anything({
      seed_candidate: 'x',
      evaluator: () => [0.5, {}],
      config: { engine: { max_metric_calls: 3 }, reflection: { reflection_lm: 'not-a-function' as never } },
    })).rejects.toThrow(/reflection_lm/);
  });
});
