import { type Candidate, type EvaluationBatch, type SideInfo } from "./types";

type ObjectiveScores = Record<string, number>;

type WrappedEvaluator = {
  call(candidate: Candidate, example?: unknown): Promise<[number, unknown, SideInfo]>;
};

function is_record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function to_objective_scores(value: unknown): ObjectiveScores {
  if (!is_record(value)) {
    return {};
  }

  const ret: ObjectiveScores = {};
  for (const [key, raw_score] of Object.entries(value)) {
    if (typeof raw_score === "number") {
      ret[key] = raw_score;
    }
  }
  return ret;
}

function extract_objective_scores(side_info: SideInfo, candidate: Candidate): ObjectiveScores {
  const objective_score: ObjectiveScores = {};

  for (const [k, v] of Object.entries(to_objective_scores(side_info.scores))) {
    objective_score[k] = v;
  }

  for (const param_name of Object.keys(candidate)) {
    const specific_info = side_info[`${param_name}_specific_info`];
    if (!is_record(specific_info)) {
      continue;
    }

    const component_scores = to_objective_scores(specific_info.scores);
    for (const [k, v] of Object.entries(component_scores)) {
      objective_score[`${param_name}::${k}`] = v;
    }
  }

  return objective_score;
}

export class OptimizeAnythingAdapter {
  private readonly evaluator: WrappedEvaluator;

  constructor(params: { evaluator: WrappedEvaluator }) {
    this.evaluator = params.evaluator;
  }

  async evaluate(batch: unknown[], candidate: Candidate, _capture_traces: boolean = false): Promise<EvaluationBatch> {
    const outputs: Array<[number, Candidate, SideInfo]> = [];
    const scores: number[] = [];
    const trajectories: SideInfo[] = [];
    const objective_scores: ObjectiveScores[] = [];

    for (const example of batch) {
      const [score, _output, side_info] = await this.evaluator.call(candidate, example);
      outputs.push([score, candidate, side_info]);
      scores.push(score);
      trajectories.push(side_info);
      objective_scores.push(extract_objective_scores(side_info, candidate));
    }

    return {
      outputs,
      scores,
      trajectories,
      objective_scores,
      num_metric_calls: batch.length,
    };
  }

  make_reflective_dataset(
    _candidate: Candidate,
    eval_batch: EvaluationBatch<SideInfo, unknown>,
    components_to_update: string[],
  ): Record<string, Array<Record<string, unknown>>> {
    const side_infos = eval_batch.trajectories ?? [];
    const ret: Record<string, Array<Record<string, unknown>>> = {};

    for (const component_name of components_to_update) {
      ret[component_name] = [];

      for (const side_info of side_infos) {
        const record: Record<string, unknown> = {};

        for (const [k, v] of Object.entries(side_info)) {
          if (k === "scores") {
            record["Scores (Higher is Better)"] = v;
          } else if (k === `${component_name}_specific_info`) {
            if (is_record(v)) {
              for (const [specific_key, specific_value] of Object.entries(v)) {
                record[specific_key] = specific_value;
              }
            }
          } else if (k.endsWith("_specific_info")) {
            continue;
          } else {
            record[k] = v;
          }
        }

        ret[component_name].push(record);
      }
    }

    return ret;
  }
}
