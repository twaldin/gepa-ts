import type { Candidate, EvaluationBatch, GEPAAdapter, ProposalFn } from '../../types.js';

export const TOOL_MODULE_PREFIX = 'tool_module';

export type DSPyPrediction = Record<string, unknown>;
export type DSPyExample = Record<string, unknown>;
export type DSPyScoreWithFeedback = {
  score?: number | null;
  feedback?: string | null;
  subscores?: Record<string, number> | null;
};

export type DSPySignature = {
  instructions?: string;
  with_instructions?: (instructions: string) => DSPySignature;
  equals?: (other: DSPySignature) => boolean;
};

export type DSPyPredictor = {
  signature?: DSPySignature;
};

export type DSPyTool = {
  name: string;
  desc?: string;
  args?: Record<string, Record<string, unknown>>;
};

export type DSPyProgram = {
  deepcopy?: () => DSPyProgram;
  named_predictors?: () => Iterable<[string, DSPyPredictor]>;
  tools?: Record<string, DSPyTool>;
  [key: string]: unknown;
};

export type DSPyTraceEntry = {
  predictor?: DSPyPredictor | string;
  inputs: Record<string, unknown>;
  prediction: DSPyPrediction | DSPyFailedPrediction;
};

export type DSPyFailedPrediction = {
  failed_prediction: true;
  completion_text: string;
};

export type DSPyTraceData = {
  trace?: DSPyTraceEntry[];
  example?: DSPyExample;
  prediction?: DSPyPrediction | DSPyFailedPrediction;
  score?: unknown;
};

export type DSPyEvaluationHook = (args: {
  batch: DSPyExample[];
  candidate: Candidate;
  program: DSPyProgram;
  capture_traces: boolean;
}) => EvaluationBatch<DSPyTraceData, DSPyPrediction | DSPyFailedPrediction> | Promise<EvaluationBatch<DSPyTraceData, DSPyPrediction | DSPyFailedPrediction>>;

export type DSPyFeedbackFn = (args: {
  predictor_output: DSPyPrediction;
  predictor_inputs: Record<string, unknown>;
  module_inputs: DSPyExample;
  module_outputs: DSPyPrediction | DSPyFailedPrediction | undefined;
  captured_trace: DSPyTraceEntry[];
}) => DSPyScoreWithFeedback | Record<string, unknown>;

export type DSPyAdapterConfig = {
  student_module: DSPyProgram;
  metric_fn: unknown;
  feedback_map: Record<string, DSPyFeedbackFn>;
  failure_score?: number;
  num_threads?: number | null;
  add_format_failure_as_feedback?: boolean;
  rng?: { choice?: <T>(items: T[]) => T };
  reflection_lm?: ((prompt: string) => string | string[] | Record<string, unknown> | Array<string | Record<string, unknown>> | Promise<string | string[] | Record<string, unknown> | Array<string | Record<string, unknown>>>) | null;
  custom_instruction_proposer?: ProposalFn | null;
  warn_on_score_mismatch?: boolean;
  enable_tool_optimization?: boolean;
  reflection_minibatch_size?: number | null;
  evaluator?: DSPyEvaluationHook | null;
};

type ExtractedScore = { score: number | null; subscores: Record<string, number> };

function is_record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function is_failed_prediction(value: unknown): value is DSPyFailedPrediction {
  return is_record(value) && value.failed_prediction === true && typeof value.completion_text === 'string';
}

function as_prediction(value: unknown): DSPyPrediction {
  return is_record(value) ? value : {};
}

function clone_program(program: DSPyProgram): DSPyProgram {
  if (typeof program.deepcopy === 'function') {
    return program.deepcopy();
  }
  return { ...program };
}

function named_predictors(program: DSPyProgram): Array<[string, DSPyPredictor]> {
  if (typeof program.named_predictors !== 'function') {
    return [];
  }
  return Array.from(program.named_predictors());
}

function stringify_value(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value === null || value === undefined) return String(value);
  if (is_record(value) || Array.isArray(value)) return JSON.stringify(value);
  return String(value);
}

function format_examples(reflective_dataset: Array<Record<string, unknown>>): string {
  const render_value = (value: unknown, level: number): string => {
    if (is_record(value)) {
      const entries = Object.entries(value);
      if (entries.length === 0) return '\n';
      return entries.map(([key, child]) => `${'#'.repeat(level)} ${key}\n${render_value(child, Math.min(level + 1, 6))}`).join('');
    }
    if (Array.isArray(value)) {
      if (value.length === 0) return '\n';
      return value.map((child, idx) => `${'#'.repeat(level)} Item ${idx + 1}\n${render_value(child, Math.min(level + 1, 6))}`).join('');
    }
    return `${stringify_value(value).trim()}\n\n`;
  };

  return reflective_dataset
    .map((sample, idx) => {
      let text = `# Example ${idx + 1}\n`;
      for (const [key, value] of Object.entries(sample)) {
        text += `## ${key}\n${render_value(value, 3)}`;
      }
      return text;
    })
    .join('\n\n');
}

export class ToolProposer {
  call(candidate: Candidate, reflective_dataset: Record<string, Array<Record<string, unknown>>>, components_to_update: string[]): Candidate {
    const updated: Candidate = {};
    for (const module_key of components_to_update) {
      const current = candidate[module_key];
      const examples = reflective_dataset[module_key];
      if (current === undefined || examples === undefined) continue;

      const parsed: unknown = JSON.parse(current);
      if (!is_record(parsed)) {
        throw new Error(`Tool module candidate ${module_key} must be a JSON object.`);
      }
      const improved: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(parsed)) {
        if (typeof value === 'string') {
          improved[key] = value;
        }
      }
      improved.tools = parsed.tools ?? {};
      updated[module_key] = JSON.stringify(improved, null, 2);
      void format_examples(examples);
    }
    return updated;
  }
}

export class InstructionProposalSignature {
  static prompt_renderer(input_dict: { current_instruction_doc: string; dataset_with_feedback: Array<Record<string, unknown>> }): string {
    return [
      'I provided an assistant with instructions to perform a task, but the assistant needs improvement based on the examples and feedback below.',
      '',
      '## Current Instruction',
      input_dict.current_instruction_doc,
      '',
      '## Examples With Feedback',
      format_examples(input_dict.dataset_with_feedback),
      '',
      'Write a better instruction. Return only the improved instruction text.',
    ].join('\n');
  }

  static output_extractor(output: string): { new_instruction: string } {
    return { new_instruction: output.trim() };
  }

  static async run(args: {
    lm: (prompt: string) => string | Promise<string>;
    input_dict: { current_instruction_doc: string; dataset_with_feedback: Array<Record<string, unknown>> };
  }): Promise<{ new_instruction: string }> {
    const prompt = this.prompt_renderer(args.input_dict);
    return this.output_extractor(await args.lm(prompt));
  }
}

export class DspyAdapter implements GEPAAdapter<DSPyExample, DSPyTraceData, DSPyPrediction | DSPyFailedPrediction> {
  readonly student: DSPyProgram;
  readonly metric_fn: unknown;
  readonly feedback_map: Record<string, DSPyFeedbackFn>;
  readonly failure_score: number;
  readonly num_threads: number | null;
  readonly add_format_failure_as_feedback: boolean;
  readonly reflection_lm: DSPyAdapterConfig['reflection_lm'];
  readonly custom_instruction_proposer: ProposalFn | null;
  readonly enable_tool_optimization: boolean;
  readonly reflection_minibatch_size: number | null;
  private readonly evaluator: DSPyEvaluationHook | null;
  private warn_on_score_mismatch: boolean;

  constructor({
    student_module,
    metric_fn,
    feedback_map,
    failure_score = 0.0,
    num_threads = null,
    add_format_failure_as_feedback = false,
    rng,
    reflection_lm = null,
    custom_instruction_proposer = null,
    warn_on_score_mismatch = true,
    enable_tool_optimization = false,
    reflection_minibatch_size = null,
    evaluator = null,
  }: DSPyAdapterConfig) {
    this.student = student_module;
    this.metric_fn = metric_fn;
    this.feedback_map = feedback_map;
    this.failure_score = failure_score;
    this.num_threads = num_threads;
    this.add_format_failure_as_feedback = add_format_failure_as_feedback;
    this.reflection_lm = reflection_lm;
    this.custom_instruction_proposer = custom_instruction_proposer;
    this.warn_on_score_mismatch = warn_on_score_mismatch;
    this.enable_tool_optimization = enable_tool_optimization;
    this.reflection_minibatch_size = reflection_minibatch_size;
    this.evaluator = evaluator;
    void rng;
  }

  async propose_new_texts(
    candidate: Candidate,
    reflective_dataset: Record<string, Array<Record<string, unknown>>>,
    components_to_update: string[],
  ): Promise<Candidate> {
    if (this.custom_instruction_proposer !== null) {
      return this.custom_instruction_proposer(candidate, reflective_dataset, components_to_update);
    }
    const tool_components = components_to_update.filter((name) => name.startsWith(TOOL_MODULE_PREFIX));
    const instruction_components = components_to_update.filter((name) => !name.startsWith(TOOL_MODULE_PREFIX));
    const results: Candidate = {};

    for (const name of instruction_components) {
      const base_instruction = candidate[name];
      const dataset_with_feedback = reflective_dataset[name];
      if (base_instruction === undefined || dataset_with_feedback === undefined) continue;
      const result = await InstructionProposalSignature.run({
        lm: async (prompt) => (await this.stripped_lm_call_async(prompt))[0] ?? '',
        input_dict: { current_instruction_doc: base_instruction, dataset_with_feedback },
      });
      results[name] = result.new_instruction;
    }

    if (tool_components.length > 0) {
      Object.assign(results, new ToolProposer().call(candidate, reflective_dataset, tool_components));
    }
    return results;
  }

  build_program(candidate: Candidate): DSPyProgram {
    const new_prog = clone_program(this.student);
    const predictor_candidates: Candidate = {};
    const tool_candidates: Record<string, Record<string, unknown>> = {};

    for (const [key, value] of Object.entries(candidate)) {
      if (key.startsWith(TOOL_MODULE_PREFIX)) {
        if (!this.enable_tool_optimization) continue;
        const parsed: unknown = JSON.parse(value);
        if (!is_record(parsed)) continue;
        for (const [pred_name, instruction] of Object.entries(parsed)) {
          if (typeof instruction === 'string') {
            predictor_candidates[pred_name] = instruction;
          }
        }
        if (is_record(parsed.tools)) {
          for (const [tool_name, tool_config] of Object.entries(parsed.tools)) {
            if (is_record(tool_config)) tool_candidates[tool_name] = tool_config;
          }
        }
      } else {
        predictor_candidates[key] = value;
      }
    }

    for (const [name, pred] of named_predictors(new_prog)) {
      const instruction = predictor_candidates[name];
      if (instruction === undefined) continue;
      if (pred.signature !== undefined && typeof pred.signature.with_instructions === 'function') {
        pred.signature = pred.signature.with_instructions(instruction);
      } else {
        pred.signature = { ...(pred.signature ?? {}), instructions: instruction };
      }
    }

    this._update_tool_descriptions(new_prog, tool_candidates);
    return new_prog;
  }

  _update_tool_descriptions(program: DSPyProgram, tool_candidates: Record<string, Record<string, unknown>>): void {
    const tools = this._collect_tools(program);
    for (const [tool_name, tool_config] of Object.entries(tool_candidates)) {
      const tool = tools[tool_name];
      if (tool === undefined) continue;
      if (typeof tool_config.desc === 'string') {
        tool.desc = tool_config.desc;
      }
      if (is_record(tool_config.args)) {
        tool.args ??= {};
        for (const [arg_name, arg_config] of Object.entries(tool_config.args)) {
          if (!is_record(arg_config) || typeof arg_config.description !== 'string') continue;
          tool.args[arg_name] ??= {};
          tool.args[arg_name].description = arg_config.description;
        }
      }
    }
  }

  _collect_tools(program: DSPyProgram): Record<string, DSPyTool> {
    const tools: Record<string, DSPyTool> = {};
    if (is_record(program.tools)) {
      for (const [name, value] of Object.entries(program.tools)) {
        if (is_record(value)) {
          const tool: DSPyTool = {
            name: typeof value.name === 'string' ? value.name : name,
          };
          if (typeof value.desc === 'string') tool.desc = value.desc;
          if (is_record(value.args)) tool.args = value.args as Record<string, Record<string, unknown>>;
          tools[name] = tool;
          program.tools[name] = tools[name];
        }
      }
    }
    return tools;
  }

  async evaluate(
    batch: DSPyExample[],
    candidate: Candidate,
    capture_traces: boolean = false,
  ): Promise<EvaluationBatch<DSPyTraceData, DSPyPrediction | DSPyFailedPrediction>> {
    const program = this.build_program(candidate);
    if (this.evaluator === null) {
      throw new Error('DSPy evaluation is not bundled in the zero-dependency TypeScript adapter; pass evaluator.');
    }
    const result = await this.evaluator({ batch, candidate, program, capture_traces });
    const objective_scores = result.objective_scores ?? result.scores.map((score) => {
      const extracted = DspyAdapter._extract_score_and_subscores(score);
      return extracted.subscores;
    });
    const normalized: EvaluationBatch<DSPyTraceData, DSPyPrediction | DSPyFailedPrediction> = {
      ...result,
      num_metric_calls: result.num_metric_calls ?? batch.length,
    };
    if (objective_scores.some((entry) => Object.keys(entry).length > 0)) {
      normalized.objective_scores = objective_scores;
    }
    return normalized;
  }

  static _extract_score_and_subscores(score_obj: unknown): ExtractedScore {
    if (score_obj === null || score_obj === undefined) return { score: null, subscores: {} };
    if (typeof score_obj === 'number') return { score: score_obj, subscores: {} };
    if (is_record(score_obj)) {
      const raw_score = score_obj.score;
      const score = typeof raw_score === 'number' ? raw_score : null;
      const subscores: Record<string, number> = {};
      if (is_record(score_obj.subscores)) {
        for (const [key, value] of Object.entries(score_obj.subscores)) {
          if (typeof value === 'number') subscores[key] = value;
        }
      }
      return { score, subscores };
    }
    const numeric = Number(score_obj);
    return Number.isFinite(numeric) ? { score: numeric, subscores: {} } : { score: null, subscores: {} };
  }

  make_reflective_dataset(
    _candidate: Candidate,
    eval_batch: EvaluationBatch<DSPyTraceData, DSPyPrediction | DSPyFailedPrediction>,
    components_to_update: string[],
  ): Record<string, Array<Record<string, unknown>>> {
    const ret: Record<string, Array<Record<string, unknown>>> = {};
    for (const pred_name of components_to_update) {
      const target_name = pred_name.startsWith(`${TOOL_MODULE_PREFIX}:`)
        ? pred_name.slice(`${TOOL_MODULE_PREFIX}:`.length)
        : pred_name;
      const items: Array<Record<string, unknown>> = [];
      for (const data of eval_batch.trajectories ?? []) {
        const trace = data.trace ?? [];
        const module_score = DspyAdapter._extract_score_and_subscores(data.score).score;
        const trace_instances = trace.filter((entry) => this.trace_matches(entry, target_name));
        const usable = this.add_format_failure_as_feedback
          ? trace_instances
          : trace_instances.filter((entry) => !is_failed_prediction(entry.prediction));
        const selected = usable.find((entry) => is_failed_prediction(entry.prediction)) ?? usable[0];
        if (selected === undefined) continue;
        const outputs = selected.prediction;
        const item: Record<string, unknown> = {
          Inputs: this.stringify_inputs(selected.inputs),
          'Generated Outputs': is_failed_prediction(outputs)
            ? `Couldn't parse the output as per the expected output format. The model's raw response was:\n\`\`\`\n${outputs.completion_text}\n\`\`\`\n`
            : this.stringify_outputs(outputs),
        };
        if (is_failed_prediction(outputs)) {
          item.Feedback = 'Your output failed to parse. Follow the expected output structure.';
        } else {
          const feedback_fn = this.feedback_map[target_name];
          if (feedback_fn === undefined) {
            throw new Error(`Missing feedback function for predictor ${target_name}.`);
          }
          const feedback = feedback_fn({
            predictor_output: outputs,
            predictor_inputs: selected.inputs,
            module_inputs: data.example ?? {},
            module_outputs: data.prediction,
            captured_trace: trace,
          });
          const feedback_record = is_record(feedback) ? feedback : {};
          const feedback_score = typeof feedback_record.score === 'number' ? feedback_record.score : null;
          item.Feedback = typeof feedback_record.feedback === 'string' ? feedback_record.feedback : '';
          if (
            module_score !== null &&
            feedback_score !== null &&
            Math.abs(feedback_score - module_score) > 1e-8 &&
            this.warn_on_score_mismatch
          ) {
            this.warn_on_score_mismatch = false;
          }
        }
        items.push(item);
      }
      if (items.length > 0) {
        ret[pred_name] = items;
      }
    }
    if (Object.keys(ret).length === 0) {
      throw new Error('No valid predictions found for any module.');
    }
    return ret;
  }

  stripped_lm_call(prompt: string): string[] {
    if (this.reflection_lm === null || this.reflection_lm === undefined) {
      throw new Error('reflection_lm is required to propose new DSPy instructions.');
    }
    const raw_outputs = this.reflection_lm(prompt);
    if (raw_outputs instanceof Promise) {
      throw new Error('Use propose_new_texts for async reflection_lm calls.');
    }
    const outputs = Array.isArray(raw_outputs) ? raw_outputs : [raw_outputs];
    return outputs.map((raw_output) => {
      if (typeof raw_output === 'string') return raw_output;
      if (is_record(raw_output) && typeof raw_output.text === 'string') return raw_output.text;
      if (is_record(raw_output) && !('text' in raw_output)) {
        throw new Error("Missing 'text' field in the output from the base LM!");
      }
      throw new Error('Unexpected output type from the base LM! Expected str or dict');
    });
  }

  private async stripped_lm_call_async(prompt: string): Promise<string[]> {
    if (this.reflection_lm === null || this.reflection_lm === undefined) {
      throw new Error('reflection_lm is required to propose new DSPy instructions.');
    }
    const raw_outputs = await this.reflection_lm(prompt);
    return this.normalize_lm_outputs(raw_outputs);
  }

  private normalize_lm_outputs(raw_outputs: string | string[] | Record<string, unknown> | Array<string | Record<string, unknown>>): string[] {
    const outputs = Array.isArray(raw_outputs) ? raw_outputs : [raw_outputs];
    return outputs.map((raw_output) => {
      if (typeof raw_output === 'string') return raw_output;
      if (is_record(raw_output) && typeof raw_output.text === 'string') return raw_output.text;
      if (is_record(raw_output) && !('text' in raw_output)) {
        throw new Error("Missing 'text' field in the output from the base LM!");
      }
      throw new Error('Unexpected output type from the base LM! Expected str or dict');
    });
  }

  private trace_matches(entry: DSPyTraceEntry, target_name: string): boolean {
    if (typeof entry.predictor === 'string') return entry.predictor === target_name;
    return true;
  }

  private stringify_inputs(inputs: Record<string, unknown>): Record<string, string> {
    const ret: Record<string, string> = {};
    for (const [key, value] of Object.entries(inputs)) {
      ret[key] = stringify_value(value);
    }
    return ret;
  }

  private stringify_outputs(outputs: DSPyPrediction): Record<string, string> {
    const ret: Record<string, string> = {};
    for (const [key, value] of Object.entries(outputs)) {
      ret[key] = stringify_value(value);
    }
    return ret;
  }
}
