import { describe, expect, it, vi } from 'vitest';
import { GEPAState, ValsetEvaluation } from '../../src/state.js';
import { ReflectiveMutationProposer } from '../../src/proposer.js';
import { EpochShuffledBatchSampler } from '../../src/batch_sampler.js';
import { ParetoCandidateSelector } from '../../src/candidate_selector.js';
import { RoundRobinReflectionComponentSelector } from '../../src/component_selector.js';
import { SeededRandom } from '../../src/utils.js';
import type { EvaluationBatch, GEPAAdapter } from '../../src/types.js';

function make_state(seed_candidate: Record<string, string> = { predictor: 'original text' }): GEPAState {
  const state = new GEPAState({
    seed_candidate,
    base_evaluation: new ValsetEvaluation({
      outputs_by_val_id: new Map([[0, null]]),
      scores_by_val_id: new Map([[0, 0.5]]),
    }),
  });
  // Engine always increments state.i before calling propose, so simulate that
  state.i = 0;
  state.full_program_trace.push({ i: state.i });
  return state;
}

function make_loader(items: unknown[] = [null]) {
  return {
    all_ids: () => items.map((_, i) => i),
    fetch: (ids: number[]) => ids.map((i) => items[i]),
    length: items.length,
  };
}

function make_mock_adapter(opts: {
  scores_before?: number[];
  scores_after?: number[];
  trajectories_before?: unknown[];
  trajectories_after?: unknown[];
  propose_new_texts?: ((c: Record<string, string>) => Record<string, string>) | null;
}): GEPAAdapter {
  const trajectories_before = opts.trajectories_before ?? [{}];
  const trajectories_after = opts.trajectories_after ?? [{}];

  const evaluate_mock = vi
    .fn<[unknown[], Record<string, string>, boolean], Promise<EvaluationBatch>>()
    .mockResolvedValueOnce({
      outputs: [{}],
      scores: opts.scores_before ?? [0.5],
      trajectories: trajectories_before,
      objective_scores: [{}],
      num_metric_calls: (opts.scores_before ?? [0.5]).length,
    })
    .mockResolvedValueOnce({
      outputs: [{}],
      scores: opts.scores_after ?? [0.7],
      trajectories: trajectories_after,
      objective_scores: [{}],
      num_metric_calls: (opts.scores_after ?? [0.7]).length,
    });

  const adapter: GEPAAdapter = {
    evaluate: evaluate_mock,
    make_reflective_dataset: vi.fn().mockReturnValue({ predictor: [{ example: 1 }] }),
  };

  if (opts.propose_new_texts !== null && opts.propose_new_texts !== undefined) {
    adapter.propose_new_texts = opts.propose_new_texts;
  }

  return adapter;
}

function make_proposer(opts: {
  adapter: GEPAAdapter;
  reflection_lm?: ((p: string) => Promise<string>) | null;
  perfect_score?: number | null;
  skip_perfect_score?: boolean;
}): ReflectiveMutationProposer {
  const rng = new SeededRandom(42);
  const loader = make_loader();

  return new ReflectiveMutationProposer({
    logger: { log: () => undefined },
    trainset: loader,
    adapter: opts.adapter,
    candidate_selector: new ParetoCandidateSelector(rng),
    module_selector: new RoundRobinReflectionComponentSelector(),
    batch_sampler: new EpochShuffledBatchSampler(1, rng),
    perfect_score: opts.perfect_score ?? null,
    skip_perfect_score: opts.skip_perfect_score ?? false,
    reflection_lm: opts.reflection_lm ?? null,
    callbacks: null,
  });
}

describe('ReflectiveMutationProposer', () => {
  it('full propose() cycle returns CandidateProposal with updated predictor', async () => {
    const adapter = make_mock_adapter({ scores_before: [0.5], scores_after: [0.7] });
    const reflection_lm = vi.fn().mockResolvedValue('```\nnew text\n```');
    const proposer = make_proposer({ adapter, reflection_lm });
    const state = make_state();

    const proposal = await proposer.propose(state);

    expect(proposal).not.toBeNull();
    expect(proposal?.candidate['predictor']).toBe('new text');
    expect(proposal?.parent_program_ids).toEqual([0]);
  });

  it('skip_perfect_score=true with all perfect scores returns null proposal', async () => {
    const adapter = make_mock_adapter({ scores_before: [1.0], scores_after: [1.0] });
    const reflection_lm = vi.fn().mockResolvedValue('```\nnew text\n```');
    const proposer = make_proposer({
      adapter,
      reflection_lm,
      perfect_score: 1.0,
      skip_perfect_score: true,
    });
    const state = make_state();

    const output = await proposer.propose_output(state);
    expect(output.proposal).toBeNull();
  });

  it('empty trajectories returns null proposal', async () => {
    const adapter = make_mock_adapter({
      scores_before: [0.5],
      trajectories_before: [],
      scores_after: [0.7],
    });
    const reflection_lm = vi.fn().mockResolvedValue('```\nnew text\n```');
    const proposer = make_proposer({ adapter, reflection_lm });
    const state = make_state();

    const output = await proposer.propose_output(state);
    expect(output.proposal).toBeNull();
  });

  it('propose_new_texts uses adapter.propose_new_texts when present', async () => {
    const custom_propose = vi.fn().mockReturnValue({ predictor: 'adapter text' });
    const adapter = make_mock_adapter({
      scores_before: [0.5],
      scores_after: [0.7],
      propose_new_texts: custom_propose,
    });
    const reflection_lm = vi.fn().mockResolvedValue('should not be called');
    const proposer = make_proposer({ adapter, reflection_lm });
    const state = make_state();

    const proposal = await proposer.propose(state);

    expect(custom_propose).toHaveBeenCalled();
    expect(reflection_lm).not.toHaveBeenCalled();
    expect(proposal?.candidate['predictor']).toBe('adapter text');
  });
});
