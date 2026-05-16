import { describe, expect, it } from 'vitest';
import { mkdtempSync, mkdirSync, realpathSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  TerminusAdapter,
  get_results,
  run_agent_tb,
} from '../../src/adapters/terminal_bench_adapter/index.js';

function canonicalPath(path: string): string {
  try {
    return realpathSync.native(path);
  } catch {
    return path;
  }
}

describe('terminal_bench_adapter', () => {
  it('runs the injected tb command with upstream-shaped arguments and writes the prompt template', async () => {
    const root = mkdtempSync(join(tmpdir(), 'gepa-tb-'));
    const prompt_template_path = join(root, 'prompt-templates', 'instruction_prompt.txt');
    const calls: Array<{ cmd: string[]; cwd: string }> = [];

    const code = await run_agent_tb({
      task_ids: ['task-a', 'task-b'],
      run_id: 'run-1',
      model_name: 'model-x',
      instruction_prompt: 'Use the terminal carefully.',
      prompt_template_path,
      n_concurrent: 3,
      command_runner: async (cmd, options) => {
        calls.push({ cmd, cwd: options.cwd });
        return 0;
      },
    });

    expect(code).toBe(0);
    expect(calls).toEqual([
      {
        cwd: canonicalPath(root),
        cmd: [
          'tb',
          'run',
          '--dataset-name',
          'terminal-bench-core',
          '--dataset-version',
          'head',
          '--agent-import-path',
          'train_terminus:TerminusWrapper',
          '--model-name',
          'model-x',
          '--run-id',
          'run-1',
          '--n-concurrent',
          '3',
          '--output-path',
          join(canonicalPath(process.cwd()), 'runs'),
          '--task-id',
          'task-a',
          '--task-id',
          'task-b',
        ],
      },
    ]);
  });

  it('reads terminal-bench results and appends the final assistant response', () => {
    const root = mkdtempSync(join(tmpdir(), 'gepa-tb-results-'));
    const logging_dir = join(root, 'runs', 'run-1', 'task-a', 'task-a.1');
    const episode_dir = join(logging_dir, 'agent-logs', 'episode-2');
    mkdirSync(episode_dir, { recursive: true });
    writeFileSync(
      join(logging_dir, 'results.json'),
      JSON.stringify({ parser_results: { a: 'passed', b: 'failed' }, is_resolved: false, failure_mode: 'timeout' }),
    );
    writeFileSync(join(episode_dir, 'debug.json'), JSON.stringify({ input: [{ role: 'user', content: 'do task' }] }));
    writeFileSync(join(episode_dir, 'response.json'), JSON.stringify({ command: 'echo done' }));

    expect(get_results({ task_id: 'task-a', run_id: 'run-1', cwd: root })).toEqual({
      success: false,
      score: 1,
      failed_reason: 'timeout',
      messages: [
        { role: 'user', content: 'do task' },
        { role: 'assistant', content: JSON.stringify({ command: 'echo done' }) },
      ],
    });
  });

  it('evaluates batches and builds the reflective dataset', async () => {
    const adapter = new TerminusAdapter({
      n_concurrent: 2,
      now: () => new Date('2026-05-15T12:34:56Z'),
      run_agent: async (args) => {
        expect(args.task_ids).toEqual(['task-a', 'task-b']);
        expect(args.run_id).toBe('temp_gepa_run_20260515123456');
        expect(args.instruction_prompt).toBe('Use bash.');
        return 0;
      },
      result_reader: ({ task_id, run_id }) => ({
        success: task_id === 'task-a',
        score: task_id === 'task-a' ? 1 : 0,
        failed_reason: task_id === 'task-a' ? 'unknown' : 'wrong-answer',
        messages: [{ role: 'user', content: `${run_id}:${task_id}` }],
      }),
    });

    const result = await adapter.evaluate(
      [
        { task_id: 'task-a', model_name: 'model-x' },
        { task_id: 'task-b', model_name: 'model-x' },
      ],
      { instruction_prompt: 'Use bash.' },
      true,
    );

    expect(result.outputs).toEqual([
      'Terminal Bench outputs are omitted. Please see runs/temp_gepa_run_20260515123456/task-a/ for detailed logging.',
      'Terminal Bench outputs are omitted. Please see runs/temp_gepa_run_20260515123456/task-b/ for detailed logging.',
    ]);
    expect(result.scores).toEqual([1, 0]);
    expect(result.trajectories?.map((trajectory) => trajectory.success)).toEqual([true, false]);

    expect(adapter.make_reflective_dataset({ instruction_prompt: 'Use bash.' }, result, ['instruction_prompt'])).toEqual({
      instruction_prompt: [
        {
          'Message History': [{ role: 'user', content: 'temp_gepa_run_20260515123456:task-a' }],
          'Instruction Prompt': 'Use bash.',
          Feedback: 'Successfully solved the task!',
        },
        {
          'Message History': [{ role: 'user', content: 'temp_gepa_run_20260515123456:task-b' }],
          'Instruction Prompt': 'Use bash.',
          Feedback: 'Failed to solve the task. Reason: wrong-answer',
        },
      ],
    });
  });
});
