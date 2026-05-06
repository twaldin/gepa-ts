import type { GEPAStateLike, Stopper } from "./types";

export class MaxMetricCallsStopper {
  readonly max_metric_calls: number;

  constructor(max_metric_calls: number) {
    this.max_metric_calls = max_metric_calls;
  }

  call(state: GEPAStateLike): boolean {
    return state.total_num_evals >= this.max_metric_calls;
  }
}

export class CompositeStopper {
  readonly stoppers: Stopper[];
  readonly mode: "any" | "all";

  constructor(...args: Array<Stopper | "any" | "all">) {
    const maybe_mode = args[args.length - 1];
    if (maybe_mode === "any" || maybe_mode === "all") {
      this.mode = maybe_mode;
      this.stoppers = args.slice(0, -1) as Stopper[];
    } else {
      this.mode = "any";
      this.stoppers = args as Stopper[];
    }
  }

  call(state: GEPAStateLike): boolean {
    if (this.mode === "any") {
      return this.stoppers.some((stopper) => stopper(state));
    }
    return this.stoppers.every((stopper) => stopper(state));
  }
}
