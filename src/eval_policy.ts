import type { DataLoader, DataId, EvaluationPolicy, GEPAStateLike, ProgramIdx } from "./types";

type EvalPolicyStateLike = GEPAStateLike & {
  program_candidates: Array<Record<string, string>>;
  prog_candidate_val_subscores?: Array<Map<DataId, number>>;
  get_program_average_val_subset(program_idx: number): [number, number];
};

type RoundRobinPolicyStateLike = GEPAStateLike & {
  prog_candidate_val_subscores: Array<Map<DataId, number>>;
  get_program_average_val_subset(program_idx: number): [number, number];
  valset_evaluations?: Map<DataId, number[]>;
};

export class FullEvaluationPolicy<TDataId extends DataId = DataId, TDataInst = unknown>
  implements EvaluationPolicy<TDataId, TDataInst>
{
  get_eval_batch(loader: DataLoader<TDataId, TDataInst>, _state: GEPAStateLike, _target_program_idx?: ProgramIdx): TDataId[] {
    return loader.all_ids();
  }

  get_best_program(state: GEPAStateLike): ProgramIdx {
    const typed_state = state as EvalPolicyStateLike;
    let best_program_idx = -1;
    let best_avg = Number.NEGATIVE_INFINITY;
    let best_coverage = -1;
    const score_maps = typed_state.prog_candidate_val_subscores;

    const count = score_maps === undefined ? typed_state.program_candidates.length : score_maps.length;
    for (let idx = 0; idx < count; idx += 1) {
      const scores = score_maps?.[idx];
      const coverage = scores === undefined ? typed_state.get_program_average_val_subset(idx)[1] : scores.size;
      let avg_score: number;
      if (scores === undefined) {
        avg_score = typed_state.get_program_average_val_subset(idx)[0];
      } else if (coverage > 0) {
        let total = 0;
        for (const score of scores.values()) {
          total += score;
        }
        avg_score = total / coverage;
      } else {
        avg_score = Number.NEGATIVE_INFINITY;
      }
      if (avg_score > best_avg || (avg_score === best_avg && coverage > best_coverage)) {
        best_program_idx = idx;
        best_avg = avg_score;
        best_coverage = coverage;
      }
    }

    return best_program_idx;
  }

  get_valset_score(program_idx: ProgramIdx, state: GEPAStateLike): number {
    const typed_state = state as EvalPolicyStateLike;
    return typed_state.get_program_average_val_subset(program_idx)[0];
  }
}

export class RoundRobinSampleEvaluationPolicy<TDataId extends DataId = DataId, TDataInst = unknown>
  implements EvaluationPolicy<TDataId, TDataInst>
{
  readonly batch_size: number;

  constructor(batch_size = 5) {
    if (!Number.isInteger(batch_size) || batch_size <= 0) {
      throw new Error("batch_size must be a positive integer");
    }
    this.batch_size = batch_size;
  }

  get_eval_batch(loader: DataLoader<TDataId, TDataInst>, state: GEPAStateLike, _target_program_idx?: ProgramIdx): TDataId[] {
    const all_ids = loader.all_ids();
    if (all_ids.length === 0) {
      return [];
    }

    const order_index = new Map<TDataId, number>();
    all_ids.forEach((val_id, idx) => {
      order_index.set(val_id, idx);
    });

    const valset_evaluations = this._valset_evaluations(state as RoundRobinPolicyStateLike);
    const ordered_ids = [...all_ids].sort((a, b) => {
      const count_a = valset_evaluations.get(a)?.length ?? 0;
      const count_b = valset_evaluations.get(b)?.length ?? 0;
      if (count_a !== count_b) {
        return count_a - count_b;
      }
      return (order_index.get(a) ?? 0) - (order_index.get(b) ?? 0);
    });

    return ordered_ids.slice(0, this.batch_size);
  }

  get_best_program(state: GEPAStateLike): ProgramIdx {
    const typed_state = state as RoundRobinPolicyStateLike;
    let best_idx = -1;
    let best_score = Number.NEGATIVE_INFINITY;
    let best_coverage = -1;

    typed_state.prog_candidate_val_subscores.forEach((scores, program_idx) => {
      const coverage = scores.size;
      let avg = Number.NEGATIVE_INFINITY;
      if (coverage > 0) {
        let total = 0;
        for (const score of scores.values()) {
          total += score;
        }
        avg = total / coverage;
      }
      if (avg > best_score || (avg === best_score && coverage > best_coverage)) {
        best_idx = program_idx;
        best_score = avg;
        best_coverage = coverage;
      }
    });

    return best_idx;
  }

  get_valset_score(program_idx: ProgramIdx, state: GEPAStateLike): number {
    const typed_state = state as RoundRobinPolicyStateLike;
    return typed_state.get_program_average_val_subset(program_idx)[0];
  }

  private _valset_evaluations(state: RoundRobinPolicyStateLike): Map<DataId, number[]> {
    if (state.valset_evaluations instanceof Map) {
      return state.valset_evaluations;
    }

    const evaluations = new Map<DataId, number[]>();
    state.prog_candidate_val_subscores.forEach((scores, program_idx) => {
      for (const val_id of scores.keys()) {
        const existing = evaluations.get(val_id);
        if (existing !== undefined) {
          existing.push(program_idx);
        } else {
          evaluations.set(val_id, [program_idx]);
        }
      }
    });
    return evaluations;
  }
}
