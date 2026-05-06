import type { Candidate, GEPAStateLike, ReflectionComponentSelector, Trajectory } from "./types";

type ComponentStateLike = GEPAStateLike & {
  named_predictor_id_to_update_next_for_program_candidate: number[];
};

type SelectorCallable = ReflectionComponentSelector & {
  select_components_to_update: ReflectionComponentSelector;
};

type SelectorConstructor = {
  (): SelectorCallable;
  new (): SelectorCallable;
};

function round_robin_impl(
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

function all_components_impl(
  _state: GEPAStateLike,
  _trajectories: Trajectory[],
  _subsample_scores: number[],
  _candidate_idx: number,
  candidate: Candidate,
): string[] {
  return Object.keys(candidate);
}

export const RoundRobinReflectionComponentSelector: SelectorConstructor = function round_robin_reflection_component_selector() {
  const callable = ((...args) => round_robin_impl(...args)) as SelectorCallable;
  callable.select_components_to_update = callable;
  return callable;
} as SelectorConstructor;

export const AllReflectionComponentSelector: SelectorConstructor = function all_reflection_component_selector() {
  const callable = ((...args) => all_components_impl(...args)) as SelectorCallable;
  callable.select_components_to_update = callable;
  return callable;
} as SelectorConstructor;
