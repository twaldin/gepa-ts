import { describe, expect, it } from 'vitest';
import { optimize_anything, FullEvaluationPolicy, RoundRobinSampleEvaluationPolicy } from '../../src/index.js';
import { ListDataLoader } from '../../src/data_loader.js';
import type { Candidate, DataLoader, EvaluationBatch, GEPAAdapter } from '../../src/types.js';

type WeightedExample = {
  id: number;
  difficulty: number;
  split: 'train' | 'val';
};

class AutoExpandingListLoader<T> extends ListDataLoader<T> {
  private readonly staged_items: T[];
  expansions = 0;

  constructor(initial_items: T[], staged_items: T[]) {
    super(initial_items);
    this.staged_items = [...staged_items];
  }

  has_pending(): boolean {
    return this.staged_items.length > 0;
  }

  add_next_if_available(): void {
    const next = this.staged_items.shift();
    if (next === undefined) {
      return;
    }
    this.add_items([next]);
    this.expansions += 1;
  }
}

class DummyAdapter implements GEPAAdapter<WeightedExample, Record<string, number>, Record<string, number>> {
  val_eval_calls = 0;

  constructor(
    private readonly val_loader: AutoExpandingListLoader<WeightedExample>,
    private readonly expand_after = 2,
  ) {}

  async evaluate(batch: WeightedExample[], candidate: Candidate, capture_traces = false): Promise<EvaluationBatch<Record<string, number>, Record<string, number>>> {
    const weight = Number(candidate['system_prompt']?.split('=').at(-1) ?? 0);
    const outputs = batch.map((item) => ({ id: item.id, weight }));
    const scores = batch.map((item) => Math.min(1, (weight + 1) / item.difficulty));

    if (batch.some((item) => item.split === 'val')) {
      this.val_eval_calls += 1;
      if (this.val_eval_calls === this.expand_after && this.val_loader.has_pending()) {
        this.val_loader.add_next_if_available();
      }
    }

    return {
      outputs,
      scores,
      ...(capture_traces ? { trajectories: scores.map((score) => ({ score })) } : {}),
    };
  }

  make_reflective_dataset(_candidate: Candidate, eval_batch: EvaluationBatch, components_to_update: string[]): Record<string, Array<Record<string, unknown>>> {
    const records = eval_batch.scores.map((score) => ({ score }));
    return Object.fromEntries(components_to_update.map((component) => [component, records]));
  }

  propose_new_texts(candidate: Candidate, _reflective_dataset: Record<string, Array<Record<string, unknown>>>, components_to_update: string[]): Candidate {
    const weight = Number(candidate['system_prompt']?.split('=').at(-1) ?? 0);
    return Object.fromEntries(components_to_update.map((component) => [component, `weight=${weight + 1}`]));
  }
}

describe('RoundRobinSampleEvaluationPolicy', () => {
  it('samples the least-evaluated validation ids and supports dynamic valset growth', async () => {
    const trainset: WeightedExample[] = [
      { id: 0, difficulty: 2, split: 'train' },
      { id: 1, difficulty: 3, split: 'train' },
      { id: 2, difficulty: 4, split: 'train' },
    ];
    const val_loader = new AutoExpandingListLoader<WeightedExample>(
      [
        { id: 0, difficulty: 3, split: 'val' },
        { id: 1, difficulty: 4, split: 'val' },
      ],
      [{ id: 2, difficulty: 5, split: 'val' }],
    );
    const adapter = new DummyAdapter(val_loader, 2);

    const result = await optimize_anything({
      seed_candidate: { system_prompt: 'weight=0' },
      dataset: trainset,
      valset: val_loader as DataLoader<number, WeightedExample>,
      adapter,
      config: {
        engine: {
          max_metric_calls: 12,
          candidate_selection_strategy: 'current_best',
          val_evaluation_policy: new RoundRobinSampleEvaluationPolicy<number, WeightedExample>(2),
        },
        reflection: {
          reflection_lm: async () => '```\nweight=1\n```',
        },
      },
    });

    expect(val_loader.expansions).toBe(1);
    expect(adapter.val_eval_calls).toBeGreaterThanOrEqual(2);

    const covered_ids = new Set<number>();
    for (const scores of result.val_subscores) {
      for (const id of scores.keys()) {
        covered_ids.add(Number(id));
      }
    }

    expect(covered_ids.has(2)).toBe(true);
    const non_seed_batch_sizes = result.val_subscores.slice(1).map((scores) => scores.size);
    expect(Math.max(...non_seed_batch_sizes)).toBeLessThanOrEqual(2);
  });

  it('rejects non-positive batch sizes', () => {
    expect(() => new RoundRobinSampleEvaluationPolicy(0)).toThrow(/positive integer/);
  });
});

describe('FullEvaluationPolicy', () => {
  it('selects the best program from evaluated val subscores and ignores unevaluated candidate shells', () => {
    const policy = new FullEvaluationPolicy();
    const state = {
      total_num_evals: 0,
      program_candidates: [{}, {}, {}],
      prog_candidate_val_subscores: [
        new Map([[0, 0.2]]),
        new Map([[0, 0.9]]),
      ],
      get_program_average_val_subset(program_idx: number): [number, number] {
        throw new Error(`unexpected fallback average for ${program_idx}`);
      },
    };

    expect(policy.get_best_program(state)).toBe(1);
  });

  it('returns -1 when no validation score maps exist, matching upstream empty-state behavior', () => {
    const policy = new FullEvaluationPolicy();
    const state = {
      total_num_evals: 0,
      program_candidates: [],
      prog_candidate_val_subscores: [],
      get_program_average_val_subset(_program_idx: number): [number, number] {
        throw new Error('unexpected average call');
      },
    };

    expect(policy.get_best_program(state)).toBe(-1);
  });
});
