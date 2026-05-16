import { mkdirSync, readdirSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import type { Candidate, EvaluationBatch, GEPAAdapter } from '../../types.js';

export type TerminalBenchTask = {
  task_id: string;
  model_name: string;
};

export type TerminalBenchMessage = {
  role: string;
  content: unknown;
};

export type TerminalBenchTrajectory = {
  messages: TerminalBenchMessage[];
  instruction_prompt: string;
  failed_reason: string;
  success: boolean;
};

export type TerminalBenchResult = {
  success: boolean;
  score: number;
  failed_reason: string;
  messages: TerminalBenchMessage[];
};

export type TerminalBenchCommandOptions = {
  cwd: string;
  env: NodeJS.ProcessEnv;
};

export type TerminalBenchCommandRunner = (
  cmd: string[],
  options: TerminalBenchCommandOptions,
) => number | Promise<number>;

export type RunAgentTBArgs = {
  task_ids: string | string[];
  run_id: string;
  model_name: string;
  instruction_prompt: string;
  dataset_name?: string;
  dataset_version?: string;
  agent_import_path?: string;
  n_concurrent?: number;
  prompt_template_path?: string;
  command_runner?: TerminalBenchCommandRunner;
};

export type GetResultsArgs = {
  task_id: string;
  run_id: string;
  cwd?: string;
};

export type TerminusAdapterConfig = {
  n_concurrent?: number;
  instruction_prompt_path?: string;
  run_agent?: (args: RunAgentTBArgs) => number | Promise<number>;
  result_reader?: (args: GetResultsArgs) => TerminalBenchResult;
  now?: () => Date;
};

function default_command_runner(cmd: string[], options: TerminalBenchCommandOptions): number {
  const [command, ...args] = cmd;
  if (command === undefined) return 1;
  const result = spawnSync(command, args, {
    cwd: options.cwd,
    env: options.env,
    stdio: 'inherit',
  });
  return typeof result.status === 'number' ? result.status : 1;
}

function task_ids_to_list(task_ids: string | string[]): string[] {
  return Array.isArray(task_ids) ? task_ids : [task_ids];
}

function format_run_timestamp(date: Date): string {
  const pad = (value: number) => String(value).padStart(2, '0');
  return (
    `${date.getUTCFullYear()}` +
    `${pad(date.getUTCMonth() + 1)}` +
    `${pad(date.getUTCDate())}` +
    `${pad(date.getUTCHours())}` +
    `${pad(date.getUTCMinutes())}` +
    `${pad(date.getUTCSeconds())}`
  );
}

function is_message_list(value: unknown): value is TerminalBenchMessage[] {
  return (
    Array.isArray(value) &&
    value.every((item) => {
      if (typeof item !== 'object' || item === null) return false;
      const record = item as Record<string, unknown>;
      return typeof record.role === 'string' && 'content' in record;
    })
  );
}

function parse_json_file(path: string): unknown {
  return JSON.parse(readFileSync(path, 'utf8')) as unknown;
}

function canonical_path(path: string): string {
  try {
    return realpathSync.native(path);
  } catch {
    return resolve(path);
  }
}

export async function run_agent_tb({
  task_ids,
  run_id,
  model_name,
  instruction_prompt,
  dataset_name = 'terminal-bench-core',
  dataset_version = 'head',
  agent_import_path = 'train_terminus:TerminusWrapper',
  n_concurrent = 6,
  prompt_template_path = 'prompt-templates/instruction_prompt.txt',
  command_runner = default_command_runner,
}: RunAgentTBArgs): Promise<number> {
  mkdirSync(dirname(prompt_template_path), { recursive: true });
  writeFileSync(prompt_template_path, instruction_prompt);

  const cwd = canonical_path(resolve(dirname(prompt_template_path), '..'));
  const cmd = [
    'tb',
    'run',
    '--dataset-name',
    dataset_name,
    '--dataset-version',
    dataset_version,
    '--agent-import-path',
    agent_import_path,
    '--model-name',
    model_name,
    '--run-id',
    run_id,
    '--n-concurrent',
    String(n_concurrent),
    '--output-path',
    join(canonical_path(process.env.PWD ?? '.'), 'runs'),
  ];
  for (const task_id of task_ids_to_list(task_ids)) {
    cmd.push('--task-id', task_id);
  }

  try {
    return await command_runner(cmd, { cwd, env: process.env });
  } catch {
    return 1;
  }
}

export function get_results({ task_id, run_id, cwd: results_cwd = process.env.PWD ?? '.' }: GetResultsArgs): TerminalBenchResult {
  const logging_dir_base = join(results_cwd, 'runs', run_id, task_id);
  const logging_dir_name = readdirSync(logging_dir_base).find((entry) => entry.startsWith(task_id));
  if (logging_dir_name === undefined) {
    throw new Error(`No logging directory found for task ${task_id} and run ${run_id}`);
  }
  const logging_dir = join(logging_dir_base, logging_dir_name);
  const result = parse_json_file(join(logging_dir, 'results.json'));
  if (typeof result !== 'object' || result === null) {
    throw new Error(`Invalid results.json for task ${task_id} and run ${run_id}`);
  }
  const result_record = result as Record<string, unknown>;
  let score = 0;
  if (typeof result_record.parser_results === 'object' && result_record.parser_results !== null) {
    score = Object.values(result_record.parser_results).filter((value) => value === 'passed').length;
  }
  const success = result_record.is_resolved === true;
  const failed_reason = typeof result_record.failure_mode === 'string' ? result_record.failure_mode : 'unknown';

  const agent_logs_dir = join(logging_dir, 'agent-logs');
  const episode_dir_name = readdirSync(agent_logs_dir)
    .filter((entry) => entry.startsWith('episode-'))
    .sort((left, right) => {
      const left_num = Number(left.split('-')[1] ?? 0);
      const right_num = Number(right.split('-')[1] ?? 0);
      return left_num - right_num;
    })
    .at(-1);
  if (episode_dir_name === undefined) {
    throw new Error(`No episode directory found for task ${task_id} and run ${run_id}`);
  }

  const episode_dir = join(agent_logs_dir, episode_dir_name);
  const trajectory = parse_json_file(join(episode_dir, 'debug.json'));
  let messages: TerminalBenchMessage[] = [];
  if (typeof trajectory === 'object' && trajectory !== null) {
    const input = (trajectory as Record<string, unknown>).input;
    if (is_message_list(input)) {
      messages = [...input];
    }
  }

  try {
    const response = parse_json_file(join(episode_dir, 'response.json'));
    messages.push({
      role: 'assistant',
      content: JSON.stringify(response),
    });
  } catch {
    // Upstream silently skips malformed or missing response.json.
  }

  return { success, score, failed_reason, messages };
}

export class TerminusAdapter implements GEPAAdapter<TerminalBenchTask, TerminalBenchTrajectory, string> {
  readonly n_concurrent: number;
  readonly instruction_prompt_path: string;
  private readonly run_agent: (args: RunAgentTBArgs) => number | Promise<number>;
  private readonly result_reader: (args: GetResultsArgs) => TerminalBenchResult;
  private readonly now: () => Date;

  constructor({
    n_concurrent = 6,
    instruction_prompt_path = 'prompt-templates/instruction_prompt.txt',
    run_agent = run_agent_tb,
    result_reader = get_results,
    now = () => new Date(),
  }: TerminusAdapterConfig = {}) {
    this.n_concurrent = n_concurrent;
    this.instruction_prompt_path = instruction_prompt_path;
    this.run_agent = run_agent;
    this.result_reader = result_reader;
    this.now = now;
  }

  async evaluate(
    batch: TerminalBenchTask[],
    candidate: Candidate,
    _capture_traces: boolean = false,
  ): Promise<EvaluationBatch<TerminalBenchTrajectory, string>> {
    if (batch.length === 0) {
      return { outputs: [], scores: [], trajectories: [], num_metric_calls: 0 };
    }
    const instruction_prompt = candidate.instruction_prompt;
    if (instruction_prompt === undefined) {
      throw new Error("Candidate must contain 'instruction_prompt'.");
    }

    const example_run_id = `temp_gepa_run_${format_run_timestamp(this.now())}`;
    const example_model_name = batch[0]!.model_name;
    await this.run_agent({
      task_ids: batch.map((task) => task.task_id),
      run_id: example_run_id,
      model_name: example_model_name,
      instruction_prompt,
      n_concurrent: this.n_concurrent,
      prompt_template_path: this.instruction_prompt_path,
    });

    const outputs: string[] = [];
    const scores: number[] = [];
    const trajectories: TerminalBenchTrajectory[] = [];
    for (const example of batch) {
      let result: TerminalBenchResult;
      try {
        result = this.result_reader({ task_id: example.task_id, run_id: example_run_id });
      } catch (error) {
        result = {
          success: false,
          score: 0,
          failed_reason: error instanceof Error ? error.message : String(error),
          messages: [],
        };
      }
      outputs.push(
        `Terminal Bench outputs are omitted. Please see runs/${example_run_id}/${example.task_id}/ for detailed logging.`,
      );
      scores.push(result.score);
      trajectories.push({
        messages: result.messages,
        instruction_prompt,
        failed_reason: result.failed_reason,
        success: result.success,
      });
    }

    return { outputs, scores, trajectories, num_metric_calls: batch.length };
  }

  make_reflective_dataset(
    candidate: Candidate,
    eval_batch: EvaluationBatch<TerminalBenchTrajectory, string>,
    _components_to_update: string[],
  ): Record<string, Array<Record<string, unknown>>> {
    const trajectories = eval_batch.trajectories ?? [];
    return {
      instruction_prompt: trajectories.map((trajectory) => ({
        'Message History': trajectory.messages,
        'Instruction Prompt': candidate.instruction_prompt,
        Feedback: trajectory.success
          ? 'Successfully solved the task!'
          : `Failed to solve the task. Reason: ${trajectory.failed_reason}`,
      })),
    };
  }
}
