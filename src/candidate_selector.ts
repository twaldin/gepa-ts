import type { CandidateSelector, GEPAStateLike } from "./types";
import { select_program_candidate_from_pareto_front, idxmax, SeededRandom } from "./utils";

type CandidateStateLike = GEPAStateLike & {
  program_at_pareto_front_valset: Map<string | number, Set<number>>;
  program_full_scores_val_set: number[];
  per_program_tracked_scores: number[];
  get_pareto_front_mapping?: () => Map<unknown, Set<number>>;
};

function key_to_record_key(key: unknown): string | number {
  if (typeof key === "string" || typeof key === "number") {
    return key;
  }
  return JSON.stringify(key);
}

function map_to_record(fronts: Map<unknown, Set<number>>): Record<string | number, Set<number>> {
  const out: Record<string | number, Set<number>> = {};
  for (const [key, value] of fronts.entries()) {
    out[key_to_record_key(key)] = new Set(value);
  }
  return out;
}

export class ParetoCandidateSelector implements CandidateSelector {
  private readonly rng: SeededRandom;

  constructor(rng: SeededRandom) {
    this.rng = rng;
  }

  select_candidate_idx(state: GEPAStateLike): number {
    const typed_state = state as CandidateStateLike;
    const pareto_front = typed_state.get_pareto_front_mapping?.() ?? typed_state.program_at_pareto_front_valset;
    return select_program_candidate_from_pareto_front(
      map_to_record(pareto_front),
      typed_state.per_program_tracked_scores,
      this.rng,
    );
  }
}

export class CurrentBestCandidateSelector implements CandidateSelector {
  select_candidate_idx(state: GEPAStateLike): number {
    const typed_state = state as CandidateStateLike;
    return idxmax(typed_state.program_full_scores_val_set);
  }
}

export class EpsilonGreedyCandidateSelector implements CandidateSelector {
  private readonly epsilon: number;
  private readonly rng: SeededRandom;

  constructor(epsilon: number, rng: SeededRandom) {
    if (epsilon < 0 || epsilon > 1) {
      throw new Error('epsilon must be between 0.0 and 1.0');
    }
    this.epsilon = epsilon;
    this.rng = rng;
  }

  select_candidate_idx(state: GEPAStateLike): number {
    const typed_state = state as CandidateStateLike;
    if (this.rng.random() < this.epsilon) {
      return this.rng.randint(0, typed_state.per_program_tracked_scores.length - 1);
    }
    return idxmax(typed_state.program_full_scores_val_set);
  }
}

export class TopKParetoCandidateSelector implements CandidateSelector {
  private readonly k: number;
  private readonly rng: SeededRandom;

  constructor(k: number, rng: SeededRandom) {
    if (!Number.isInteger(k) || k <= 0) {
      throw new Error('k must be a positive integer');
    }
    this.k = k;
    this.rng = rng;
  }

  select_candidate_idx(state: GEPAStateLike): number {
    const typed_state = state as CandidateStateLike;
    const scores = typed_state.per_program_tracked_scores;
    const pareto_front = typed_state.get_pareto_front_mapping?.() ?? typed_state.program_at_pareto_front_valset;
    const top_k_indices = new Set(
      scores
        .map((score, idx) => ({ score, idx }))
        .sort((a, b) => b.score - a.score)
        .slice(0, this.k)
        .map(({ idx }) => idx),
    );

    const filtered_fronts = new Map<unknown, Set<number>>();
    for (const [val_id, front] of pareto_front.entries()) {
      const filtered = new Set([...front].filter((program_idx) => top_k_indices.has(program_idx)));
      if (filtered.size > 0) {
        filtered_fronts.set(val_id, filtered);
      }
    }

    if (filtered_fronts.size === 0) {
      return idxmax(scores);
    }

    return select_program_candidate_from_pareto_front(
      map_to_record(filtered_fronts),
      scores,
      this.rng,
    );
  }
}
