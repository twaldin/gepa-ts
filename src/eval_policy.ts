import type { DataLoader, DataId, EvaluationPolicy, GEPAStateLike, ProgramIdx } from "./types";

type EvalPolicyStateLike = GEPAStateLike & {
  program_candidates: Array<Record<string, string>>;
  get_program_average_val_subset(program_idx: number): [number, number];
};

export class FullEvaluationPolicy<TDataId extends DataId = DataId, TDataInst = unknown>
  implements EvaluationPolicy<TDataId, TDataInst>
{
  get_eval_batch(loader: DataLoader<TDataId, TDataInst>, _state: GEPAStateLike, _target_program_idx?: ProgramIdx): TDataId[] {
    return loader.all_ids();
  }

  get_best_program(state: GEPAStateLike): ProgramIdx {
    const typed_state = state as EvalPolicyStateLike;
    let best_program_idx = 0;
    let best_avg = Number.NEGATIVE_INFINITY;
    let best_coverage = -1;

    for (let idx = 0; idx < typed_state.program_candidates.length; idx += 1) {
      const [avg_score, coverage] = typed_state.get_program_average_val_subset(idx);
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
