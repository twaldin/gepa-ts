import type { GEPAStateLike, Stopper } from "./types";

type CallableMaxMetricCallsStopper = MaxMetricCallsStopper & Stopper;
type CallableCompositeStopper = CompositeStopper & Stopper;

export class MaxMetricCallsStopper {
  readonly max_metric_calls: number;

  constructor(max_metric_calls: number) {
    this.max_metric_calls = max_metric_calls;

    const callable: CallableMaxMetricCallsStopper = Object.assign(
      (state: GEPAStateLike) => state.total_num_evals >= max_metric_calls,
      { max_metric_calls },
    );
    Object.setPrototypeOf(callable, MaxMetricCallsStopper.prototype);
    return callable;
  }
}

export class CompositeStopper {
  readonly stoppers: Stopper[];
  readonly mode: "any" | "all";

  constructor(...args: Array<Stopper | "any" | "all">) {
    const maybe_mode = args.at(-1);
    const has_explicit_mode = maybe_mode === "any" || maybe_mode === "all";
    const mode: "any" | "all" = has_explicit_mode ? maybe_mode : "any";
    const stoppers = (has_explicit_mode ? args.slice(0, -1) : args) as Stopper[];

    this.mode = mode;
    this.stoppers = stoppers;

    const callable: CallableCompositeStopper = Object.assign(
      (state: GEPAStateLike) => {
        if (mode === "any") {
          return stoppers.some((stopper) => stopper(state));
        }
        return stoppers.every((stopper) => stopper(state));
      },
      {
        mode,
        stoppers,
      },
    );
    Object.setPrototypeOf(callable, CompositeStopper.prototype);
    return callable;
  }
}
