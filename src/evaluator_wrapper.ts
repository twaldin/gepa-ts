import { STR_CANDIDATE_KEY, type Candidate, type Evaluator, type SideInfo } from "./types";

export class EvaluatorWrapper {
  private readonly evaluator: Evaluator;
  private readonly single_instance_mode: boolean;
  private readonly str_candidate_mode: boolean;
  private readonly raise_on_exception: boolean;

  constructor(
    evaluator: Evaluator,
    single_instance_mode: boolean,
    str_candidate_mode: boolean = false,
    raise_on_exception: boolean = true,
  ) {
    this.evaluator = evaluator;
    this.single_instance_mode = single_instance_mode;
    this.str_candidate_mode = str_candidate_mode;
    this.raise_on_exception = raise_on_exception;
  }

  async call(candidate: Candidate, example?: unknown): Promise<[number, unknown, SideInfo]> {
    const eval_candidate: string | Candidate = this.str_candidate_mode ? (candidate[STR_CANDIDATE_KEY] ?? "") : candidate;

    try {
      const result = this.single_instance_mode
        ? await Promise.resolve(this.evaluator(eval_candidate))
        : await Promise.resolve(this.evaluator(eval_candidate, { example }));

      if (typeof result === "number") {
        return [result, undefined, {}];
      }
      return [result[0], undefined, result[1] ?? {}];
    } catch (error) {
      if (this.raise_on_exception) {
        throw error;
      }
      return [0.0, undefined, { error: String(error) }];
    }
  }
}
