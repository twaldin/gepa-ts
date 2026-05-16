import { describe, expect, test } from 'vitest';
import {
  ConfidenceAdapter,
  LinearBlendScoring,
  ThresholdScoring,
  _build_feedback,
  _extract_answer_from_json,
} from '../../src/adapters/confidence_adapter/index.js';

function response(content: string): { choices: Array<{ message: { content: string } }> } {
  return { choices: [{ message: { content } }] };
}

const sample = [
  { input: 'UBER EATS payment', additional_context: {}, answer: 'Food & Drinks/Restaurants' },
  { input: 'LIGHT electricity bill', additional_context: { merchant_type: 'utility' }, answer: 'Bills/Electricity' },
];

describe('ConfidenceAdapter helpers', () => {
  test('extracts simple and nested JSON answer fields', () => {
    expect(_extract_answer_from_json(JSON.stringify({ category_name: 'Bills/Electricity' }), 'category_name')).toBe('Bills/Electricity');
    expect(_extract_answer_from_json(JSON.stringify({ classification: { name: 'Shopping' } }), 'classification.name')).toBe('Shopping');
    expect(_extract_answer_from_json('not json', 'category_name')).toBeNull();
  });

  test('builds confidence-aware feedback', () => {
    expect(_build_feedback({
      is_correct: true,
      expected: 'Bills/Electricity',
      got: 'Bills/Electricity',
      logprob_score: -0.01,
      top_alternatives: [],
      additional_context: {},
      high_confidence_prob: 0.99,
      low_confidence_prob: 0.90,
    })).toBe('Correct.');

    const low_confidence = _build_feedback({
      is_correct: true,
      expected: 'Bills/Electricity',
      got: 'Bills/Electricity',
      logprob_score: -2.3,
      top_alternatives: [{ token: 'gas', probability: 0.09, resolved_value: 'Bills/Gas & Oil' }],
      additional_context: {},
      high_confidence_prob: 0.99,
      low_confidence_prob: 0.90,
    });
    expect(low_confidence).toContain('uncertain');
    expect(low_confidence).toContain('Bills/Gas & Oil');

    const wrong = _build_feedback({
      is_correct: false,
      expected: 'Shopping/Video Games',
      got: 'Shopping/Electronics',
      logprob_score: -0.005,
      top_alternatives: [],
      additional_context: { merchant_type: 'retail' },
      high_confidence_prob: 0.99,
      low_confidence_prob: 0.50,
    });
    expect(wrong).toContain('WRONG');
    expect(wrong).toContain('merchant_type: retail');
  });
});

describe('ConfidenceAdapter scoring strategies', () => {
  test('linear blend and threshold strategies match upstream shape', () => {
    const linear = new LinearBlendScoring(0.5, 0.3);
    expect(linear.score(true, null)).toBe(1);
    expect(linear.score(false, -0.001)).toBe(0);
    expect(linear.score(true, -2)).toBeGreaterThan(0);
    expect(linear.score(true, -2)).toBeLessThan(1);

    const threshold = new ThresholdScoring(0.7);
    expect(threshold.score(true, -0.51)).toBe(0);
    expect(threshold.score(true, -0.001)).toBe(1);
  });
});

describe('ConfidenceAdapter', () => {
  test('evaluates callable model responses with objective accuracy and probability', async () => {
    const adapter = new ConfidenceAdapter({
      model: async () => response(JSON.stringify({ category_name: 'Food & Drinks/Restaurants' })),
      field_path: 'category_name',
      logprob_extractor: async () => [{ joint_logprob: -0.001, top_logprobs: [] }],
    });

    const result = await adapter.evaluate([sample[0]!], { system_prompt: 'Classify.' });

    expect(result.scores).toEqual([1]);
    expect(result.objective_scores?.[0]?.accuracy).toBe(1);
    expect(result.objective_scores?.[0]?.probability).toBeCloseTo(Math.exp(-0.001));
    expect(result.outputs[0]?.parsed_value).toBe('Food & Drinks/Restaurants');
    expect(result.trajectories).toBeUndefined();
  });

  test('penalizes low-confidence correct answers with custom linear scoring', async () => {
    const adapter = new ConfidenceAdapter({
      model: async () => response(JSON.stringify({ category_name: 'Bills/Electricity' })),
      field_path: 'category_name',
      scoring_strategy: new LinearBlendScoring(0.5, 0.3),
      logprob_extractor: async () => [{ joint_logprob: -2.0 }],
    });

    const result = await adapter.evaluate([sample[1]!], { system_prompt: 'Classify.' });

    expect(result.scores[0]).toBeGreaterThan(0);
    expect(result.scores[0]).toBeLessThan(1);
  });

  test('captures traces and builds reflective dataset with confidence feedback', async () => {
    const adapter = new ConfidenceAdapter({
      model: async () => response(JSON.stringify({ category_name: 'Bills/Electricity' })),
      field_path: 'category_name',
      logprob_extractor: async () => [{
        joint_logprob: -1.14,
        top_logprobs: [{ token: 'gas', probability: 0.09, resolved_value: 'Bills/Gas & Oil' }],
      }],
    });

    const eval_batch = await adapter.evaluate([sample[1]!], { system_prompt: 'Classify.' }, true);
    const dataset = adapter.make_reflective_dataset({ system_prompt: 'Classify.' }, eval_batch, ['system_prompt']);

    expect(eval_batch.trajectories?.[0]?.is_correct).toBe(true);
    expect(dataset.system_prompt?.[0]?.Inputs).toBe('LIGHT electricity bill');
    expect(dataset.system_prompt?.[0]?.['Generated Outputs']).toContain('probability');
    expect(dataset.system_prompt?.[0]?.Feedback).toContain('Bills/Gas & Oil');
  });

  test('returns failure score on model errors and requires traces for reflection', async () => {
    const adapter = new ConfidenceAdapter({
      model: async () => {
        throw new Error('API timeout');
      },
      field_path: 'category_name',
      failure_score: 0,
    });

    const eval_batch = await adapter.evaluate([sample[0]!], { system_prompt: 'Classify.' });

    expect(eval_batch.scores).toEqual([0]);
    expect(eval_batch.outputs[0]?.parsed_value).toBeNull();
    expect(() => adapter.make_reflective_dataset({ system_prompt: 'Classify.' }, eval_batch, ['system_prompt'])).toThrow('Trajectories are required');
  });
});
