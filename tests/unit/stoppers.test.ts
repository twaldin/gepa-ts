import { describe, expect, it } from 'vitest';
import {
  CompositeStopper,
  MaxCandidateProposalsStopper,
  MaxMetricCallsStopper,
  MaxReflectionCostStopper,
  MaxTrackedCandidatesStopper,
  NoImprovementStopper,
  ScoreThresholdStopper,
  SignalStopper,
  TimeoutStopCondition,
} from '../../src/index.js';

describe('upstream-compatible stoppers', () => {
  it('stops when metric calls reach the configured maximum', () => {
    const stopper = new MaxMetricCallsStopper(3);

    expect(stopper({ total_num_evals: 2 })).toBe(false);
    expect(stopper({ total_num_evals: 3 })).toBe(true);
  });

  it('stops when the best validation score reaches the threshold', () => {
    const stopper = new ScoreThresholdStopper(0.8);

    expect(stopper({ total_num_evals: 0, program_full_scores_val_set: [0.3, 0.7] })).toBe(false);
    expect(stopper({ total_num_evals: 0, program_full_scores_val_set: [0.3, 0.8] })).toBe(true);
  });

  it('tracks consecutive calls without score improvement and can reset the counter', () => {
    const stopper = new NoImprovementStopper(2);

    expect(stopper({ total_num_evals: 0, program_full_scores_val_set: [0.4] })).toBe(false);
    expect(stopper.best_score).toBe(0.4);
    expect(stopper.iterations_without_improvement).toBe(0);

    expect(stopper({ total_num_evals: 0, program_full_scores_val_set: [0.4] })).toBe(false);
    expect(stopper.iterations_without_improvement).toBe(1);

    expect(stopper({ total_num_evals: 0, program_full_scores_val_set: [0.4] })).toBe(true);
    stopper.reset();
    expect(stopper.iterations_without_improvement).toBe(0);
  });

  it('stops when the tracked candidate count reaches the configured maximum', () => {
    const stopper = new MaxTrackedCandidatesStopper(2);

    expect(stopper({ total_num_evals: 0, program_candidates: [{}, {}] })).toBe(true);
    expect(stopper({ total_num_evals: 0, program_candidates: [{}] })).toBe(false);
  });

  it('stops after the configured number of candidate proposals using upstream state.i semantics', () => {
    const stopper = new MaxCandidateProposalsStopper(3);

    expect(stopper({ total_num_evals: 0, i: 1 })).toBe(false);
    expect(stopper({ total_num_evals: 0, i: 2 })).toBe(true);
  });

  it('stops when reflection LM total_cost reaches the configured budget', () => {
    const reflection_lm = { total_cost: 0.2 };
    const stopper = new MaxReflectionCostStopper(0.3, reflection_lm);

    expect(stopper({ total_num_evals: 0 })).toBe(false);
    reflection_lm.total_cost = 0.3;
    expect(stopper({ total_num_evals: 0 })).toBe(true);
  });

  it('stops after the timeout has elapsed', async () => {
    const stopper = new TimeoutStopCondition(0.001);

    expect(stopper({ total_num_evals: 0 })).toBe(false);
    await new Promise((resolve) => setTimeout(resolve, 5));
    expect(stopper({ total_num_evals: 0 })).toBe(true);
  });

  it('exposes a signal stopper with cleanup without requiring a real signal', () => {
    const stopper = new SignalStopper([]);

    expect(stopper.signals).toEqual([]);
    expect(stopper.stop_requested).toBe(false);
    expect(stopper({ total_num_evals: 0 })).toBe(false);
    expect(() => stopper.cleanup()).not.toThrow();
  });

  it('combines stoppers with any/all modes', () => {
    const false_stopper = () => false;
    const true_stopper = () => true;

    expect(new CompositeStopper(false_stopper, true_stopper)({ total_num_evals: 0 })).toBe(true);
    expect(new CompositeStopper(false_stopper, true_stopper, 'all')({ total_num_evals: 0 })).toBe(false);
    expect(new CompositeStopper(true_stopper, true_stopper, 'all')({ total_num_evals: 0 })).toBe(true);
  });
});
