import type { CandidateSelector, GEPAStateLike } from "./types";
import { select_program_candidate_from_pareto_front, idxmax, SeededRandom } from "./utils";

type CandidateStateLike = GEPAStateLike & {
  program_at_pareto_front_valset: Map<string | number, Set<number>>;
  per_program_tracked_scores: number[];
};

function map_to_record(fronts: Map<string | number, Set<number>>): Record<string | number, Set<number>> {
  const out: Record<string | number, Set<number>> = {};
  for (const [key, value] of fronts.entries()) {
    out[key] = new Set(value);
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
    return select_program_candidate_from_pareto_front(
      map_to_record(typed_state.program_at_pareto_front_valset),
      typed_state.per_program_tracked_scores,
      this.rng,
    );
  }
}

export class CurrentBestCandidateSelector implements CandidateSelector {
  select_candidate_idx(state: GEPAStateLike): number {
    const typed_state = state as CandidateStateLike;
    return idxmax(typed_state.per_program_tracked_scores);
  }
}
