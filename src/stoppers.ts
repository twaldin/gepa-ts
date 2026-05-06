import type { GEPAStateLike, Stopper } from "./types";

export interface MaxMetricCallsStopper extends Stopper {
  readonly max_metric_calls: number;
}

class MaxMetricCallsStopperImpl {
  readonly max_metric_calls: number;

  constructor(max_metric_calls: number) {
    this.max_metric_calls = max_metric_calls;

    const callable: MaxMetricCallsStopper = Object.assign(
      (state: GEPAStateLike) => state.total_num_evals >= max_metric_calls,
      { max_metric_calls },
    );
    Object.setPrototypeOf(callable, MaxMetricCallsStopperImpl.prototype);
    return callable;
  }
}

export const MaxMetricCallsStopper: {
  new (max_metric_calls: number): MaxMetricCallsStopper;
  prototype: MaxMetricCallsStopper;
} = MaxMetricCallsStopperImpl as {
  new (max_metric_calls: number): MaxMetricCallsStopper;
  prototype: MaxMetricCallsStopper;
};

export interface CompositeStopper extends Stopper {
  readonly stoppers: Stopper[];
  readonly mode: "any" | "all";
}

class CompositeStopperImpl {
  readonly stoppers: Stopper[];
  readonly mode: "any" | "all";

  constructor(...args: Array<Stopper | "any" | "all">) {
    const maybe_mode = args.at(-1);
    const has_explicit_mode = maybe_mode === "any" || maybe_mode === "all";
    const mode: "any" | "all" = has_explicit_mode ? maybe_mode : "any";
    const stoppers = (has_explicit_mode ? args.slice(0, -1) : args) as Stopper[];

    this.mode = mode;
    this.stoppers = stoppers;

    const callable: CompositeStopper = Object.assign(
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
    Object.setPrototypeOf(callable, CompositeStopperImpl.prototype);
    return callable;
  }
}

export const CompositeStopper: {
  new (...args: Array<Stopper | "any" | "all">): CompositeStopper;
  prototype: CompositeStopper;
} = CompositeStopperImpl as {
  new (...args: Array<Stopper | "any" | "all">): CompositeStopper;
  prototype: CompositeStopper;
};
