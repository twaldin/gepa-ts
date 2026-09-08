import type { GEPAStateLike, Stopper } from "./types";
import { existsSync, unlinkSync } from "node:fs";

type ScoreStateLike = GEPAStateLike & {
  program_full_scores_val_set?: number[];
};

type CandidateStateLike = GEPAStateLike & {
  program_candidates?: unknown[];
};

type NodeSignal = "SIGINT" | "SIGTERM";

function current_best_score(state: GEPAStateLike): number {
  const scores = (state as ScoreStateLike).program_full_scores_val_set;
  if (!Array.isArray(scores) || scores.length === 0) {
    return 0;
  }
  return Math.max(...scores);
}

export interface TimeoutStopCondition extends Stopper {
  readonly timeout_seconds: number;
  readonly start_time_ms: number;
}

class TimeoutStopConditionImpl {
  readonly timeout_seconds: number;
  readonly start_time_ms: number;

  constructor(timeout_seconds: number) {
    this.timeout_seconds = timeout_seconds;
    this.start_time_ms = Date.now();

    const callable: TimeoutStopCondition = Object.assign(
      (_state: GEPAStateLike) => Date.now() - this.start_time_ms > timeout_seconds * 1000,
      { timeout_seconds, start_time_ms: this.start_time_ms },
    );
    Object.setPrototypeOf(callable, TimeoutStopConditionImpl.prototype);
    return callable;
  }
}

export const TimeoutStopCondition: {
  new (timeout_seconds: number): TimeoutStopCondition;
  prototype: TimeoutStopCondition;
} = TimeoutStopConditionImpl as {
  new (timeout_seconds: number): TimeoutStopCondition;
  prototype: TimeoutStopCondition;
};

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

export interface FileStopper extends Stopper {
  readonly stop_file_path: string;
  remove_stop_file(): void;
}

class FileStopperImpl {
  readonly stop_file_path: string;

  constructor(stop_file_path: string) {
    this.stop_file_path = stop_file_path;

    const callable: FileStopper = Object.assign(
      (_state: GEPAStateLike) => existsSync(stop_file_path),
      {
        stop_file_path,
        remove_stop_file: () => {
          if (existsSync(stop_file_path)) {
            unlinkSync(stop_file_path);
          }
        },
      },
    );
    Object.setPrototypeOf(callable, FileStopperImpl.prototype);
    return callable;
  }
}

export const FileStopper: {
  new (stop_file_path: string): FileStopper;
  prototype: FileStopper;
} = FileStopperImpl as {
  new (stop_file_path: string): FileStopper;
  prototype: FileStopper;
};

export interface ScoreThresholdStopper extends Stopper {
  readonly threshold: number;
}

class ScoreThresholdStopperImpl {
  readonly threshold: number;

  constructor(threshold: number) {
    this.threshold = threshold;

    const callable: ScoreThresholdStopper = Object.assign(
      (state: GEPAStateLike) => current_best_score(state) >= threshold,
      { threshold },
    );
    Object.setPrototypeOf(callable, ScoreThresholdStopperImpl.prototype);
    return callable;
  }
}

export const ScoreThresholdStopper: {
  new (threshold: number): ScoreThresholdStopper;
  prototype: ScoreThresholdStopper;
} = ScoreThresholdStopperImpl as {
  new (threshold: number): ScoreThresholdStopper;
  prototype: ScoreThresholdStopper;
};

export interface NoImprovementStopper extends Stopper {
  readonly max_iterations_without_improvement: number;
  best_score: number;
  iterations_without_improvement: number;
  reset(): void;
}

class NoImprovementStopperImpl {
  readonly max_iterations_without_improvement: number;
  best_score = Number.NEGATIVE_INFINITY;
  iterations_without_improvement = 0;

  constructor(max_iterations_without_improvement: number) {
    this.max_iterations_without_improvement = max_iterations_without_improvement;

    const callable: NoImprovementStopper = Object.assign(
      (state: GEPAStateLike) => {
        const score = current_best_score(state);
        if (score > callable.best_score) {
          callable.best_score = score;
          callable.iterations_without_improvement = 0;
        } else {
          callable.iterations_without_improvement += 1;
        }
        return callable.iterations_without_improvement >= max_iterations_without_improvement;
      },
      {
        max_iterations_without_improvement,
        best_score: this.best_score,
        iterations_without_improvement: this.iterations_without_improvement,
        reset: () => {
          callable.iterations_without_improvement = 0;
        },
      },
    );
    Object.setPrototypeOf(callable, NoImprovementStopperImpl.prototype);
    return callable;
  }
}

export const NoImprovementStopper: {
  new (max_iterations_without_improvement: number): NoImprovementStopper;
  prototype: NoImprovementStopper;
} = NoImprovementStopperImpl as {
  new (max_iterations_without_improvement: number): NoImprovementStopper;
  prototype: NoImprovementStopper;
};

export interface SignalStopper extends Stopper {
  readonly signals: NodeSignal[];
  readonly stop_requested: boolean;
  cleanup(): void;
}

class SignalStopperImpl {
  readonly signals: NodeSignal[];

  constructor(signals: NodeSignal[] = ["SIGINT", "SIGTERM"]) {
    this.signals = [...signals];
    let stop_requested = false;
    const handlers: Array<[NodeSignal, () => void]> = [];

    const callable: SignalStopper = Object.assign(
      (_state: GEPAStateLike) => stop_requested,
      {
        signals: this.signals,
        get stop_requested() {
          return stop_requested;
        },
        cleanup: () => {
          for (const [signal, handler] of handlers) {
            process.off(signal, handler);
          }
          handlers.length = 0;
        },
      },
    );

    for (const signal of this.signals) {
      const handler = () => {
        stop_requested = true;
      };
      process.on(signal, handler);
      handlers.push([signal, handler]);
    }

    Object.setPrototypeOf(callable, SignalStopperImpl.prototype);
    return callable;
  }
}

export const SignalStopper: {
  new (signals?: NodeSignal[]): SignalStopper;
  prototype: SignalStopper;
} = SignalStopperImpl as {
  new (signals?: NodeSignal[]): SignalStopper;
  prototype: SignalStopper;
};

export interface MaxTrackedCandidatesStopper extends Stopper {
  readonly max_tracked_candidates: number;
}

class MaxTrackedCandidatesStopperImpl {
  readonly max_tracked_candidates: number;

  constructor(max_tracked_candidates: number) {
    this.max_tracked_candidates = max_tracked_candidates;

    const callable: MaxTrackedCandidatesStopper = Object.assign(
      (state: GEPAStateLike) => ((state as CandidateStateLike).program_candidates?.length ?? 0) >= max_tracked_candidates,
      { max_tracked_candidates },
    );
    Object.setPrototypeOf(callable, MaxTrackedCandidatesStopperImpl.prototype);
    return callable;
  }
}

export const MaxTrackedCandidatesStopper: {
  new (max_tracked_candidates: number): MaxTrackedCandidatesStopper;
  prototype: MaxTrackedCandidatesStopper;
} = MaxTrackedCandidatesStopperImpl as {
  new (max_tracked_candidates: number): MaxTrackedCandidatesStopper;
  prototype: MaxTrackedCandidatesStopper;
};

export interface MaxCandidateProposalsStopper extends Stopper {
  readonly max_proposals: number;
}

class MaxCandidateProposalsStopperImpl {
  readonly max_proposals: number;

  constructor(max_proposals: number) {
    this.max_proposals = max_proposals;

    const callable: MaxCandidateProposalsStopper = Object.assign(
      (state: GEPAStateLike) => (state.i ?? -1) >= max_proposals - 1,
      { max_proposals },
    );
    Object.setPrototypeOf(callable, MaxCandidateProposalsStopperImpl.prototype);
    return callable;
  }
}

export const MaxCandidateProposalsStopper: {
  new (max_proposals: number): MaxCandidateProposalsStopper;
  prototype: MaxCandidateProposalsStopper;
} = MaxCandidateProposalsStopperImpl as {
  new (max_proposals: number): MaxCandidateProposalsStopper;
  prototype: MaxCandidateProposalsStopper;
};

export interface MaxReflectionCostStopper extends Stopper {
  readonly max_reflection_cost_usd: number;
  readonly reflection_lm: object | null;
}

class MaxReflectionCostStopperImpl {
  readonly max_reflection_cost_usd: number;
  readonly reflection_lm: object | null;

  constructor(max_reflection_cost_usd: number, reflection_lm: object | null = null) {
    this.max_reflection_cost_usd = max_reflection_cost_usd;
    this.reflection_lm = reflection_lm;

    const callable: MaxReflectionCostStopper = Object.assign(
      (_state: GEPAStateLike) => {
        const can_have_cost =
          reflection_lm !== null && (typeof reflection_lm === "object" || typeof reflection_lm === "function");
        const cost = can_have_cost && "total_cost" in reflection_lm
          ? (reflection_lm as { total_cost?: unknown }).total_cost
          : 0;
        return typeof cost === "number" && cost >= max_reflection_cost_usd;
      },
      {
        max_reflection_cost_usd,
        reflection_lm,
      },
    );
    Object.setPrototypeOf(callable, MaxReflectionCostStopperImpl.prototype);
    return callable;
  }
}

export const MaxReflectionCostStopper: {
  new (max_reflection_cost_usd: number, reflection_lm?: object | null): MaxReflectionCostStopper;
  prototype: MaxReflectionCostStopper;
} = MaxReflectionCostStopperImpl as {
  new (max_reflection_cost_usd: number, reflection_lm?: object | null): MaxReflectionCostStopper;
  prototype: MaxReflectionCostStopper;
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
