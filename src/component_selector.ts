import type { Candidate, GEPAStateLike, Trajectory } from "./types";

type ComponentStateLike = GEPAStateLike & {
  named_predictor_id_to_update_next_for_program_candidate: number[];
};

export class RoundRobinReflectionComponentSelector {
  select_components_to_update(
    state: GEPAStateLike,
    _trajectories: Trajectory[],
    _subsample_scores: number[],
    candidate_idx: number,
    candidate: Candidate,
  ): string[] {
    const typed_state = state as ComponentStateLike;
    const predictor_names = Object.keys(candidate);
    if (predictor_names.length === 0) {
      return [];
    }

    const current = typed_state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx] ?? 0;
    const next_component = predictor_names[current % predictor_names.length];
    typed_state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx] = (current + 1) % predictor_names.length;
    return next_component ? [next_component] : [];
  }
}

export class AllReflectionComponentSelector {
  select_components_to_update(
    _state: GEPAStateLike,
    _trajectories: Trajectory[],
    _subsample_scores: number[],
    _candidate_idx: number,
    candidate: Candidate,
  ): string[] {
    return Object.keys(candidate);
  }
}
