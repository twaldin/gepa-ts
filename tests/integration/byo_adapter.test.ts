import { describe, expect, it, vi } from 'vitest';
import { optimize_anything, type EvaluationBatch, type Evaluator } from '../../src/index.js';
import { OptimizeAnythingAdapter } from '../../src/adapter.js';

describe('optimize_anything BYO adapter contract path', () => {
  it('invokes adapter.evaluate and adapter.make_reflective_dataset end-to-end', async () => {
    const evaluateSpy = vi
      .spyOn(OptimizeAnythingAdapter.prototype, 'evaluate')
      .mockImplementation(async (batch): Promise<EvaluationBatch<Record<string, unknown>, string>> => {
        const count = batch.length;
        return {
          outputs: Array.from({ length: count }, () => 'ok'),
          scores: Array.from({ length: count }, () => 0.5),
          trajectories: Array.from({ length: count }, () => ({ trace: 't' })),
          side_infos: Array.from({ length: count }, () => ({ feedback: 'be better' })),
          num_metric_calls: count,
        };
      });

    const reflectiveDataset = {
      instructions: [
        {
          inputs: { q: 'x' },
          outputs: 'y',
          feedback: 'be better',
        },
      ],
    };

    const reflectiveSpy = vi
      .spyOn(OptimizeAnythingAdapter.prototype, 'make_reflective_dataset')
      .mockReturnValue(reflectiveDataset);

    const evaluator: Evaluator = (_candidate, _ctx) => [0.1, { note: 'typed evaluator accepted' }];
    const reflection_lm = async (_prompt: string) => '```\nimproved instructions\n```';

    try {
      const result = await optimize_anything({
        seed_candidate: { instructions: 'initial' },
        evaluator,
        dataset: [{ q: 'x' }],
        config: {
          engine: { max_metric_calls: 4 },
          reflection: { reflection_lm, reflection_minibatch_size: 1 },
        },
      });

      expect(evaluateSpy).toHaveBeenCalled();
      expect(reflectiveSpy).toHaveBeenCalled();
      expect(result.candidates.length).toBeGreaterThanOrEqual(1);
      expect(result.best_candidate).toBeDefined();
    } finally {
      evaluateSpy.mockRestore();
      reflectiveSpy.mockRestore();
    }
  });
});
