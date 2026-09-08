import { describe, expect, it } from 'vitest';
import {
  CodeExecutionResult,
  ExecutionMode,
  execute_code,
  get_code_hash,
} from '../../src/utils/index.js';

describe('code execution utilities', () => {
  it('executes JavaScript in-process and captures output and variables', () => {
    const result = execute_code('const doubled = input_value * 2; console.log("done", doubled);', {
      global_vars: { input_value: 21 },
      capture_variables: ['doubled'],
    });

    expect(result).toBeInstanceOf(CodeExecutionResult);
    expect(result.success).toBe(true);
    expect(result.stdout).toBe('done 42\n');
    expect(result.get_variable('doubled')).toBe(42);
    expect(result.to_side_info_dict()).toEqual({ Stdout: 'done 42\n', Stderr: '' });
    expect(result.code_hash).toHaveLength(64);
  });

  it('calls an entry point and exposes __return__', () => {
    const result = execute_code('function solve(x) { return x * 3; }', {
      entry_point: 'solve',
      entry_point_args: [7],
    });

    expect(result.success).toBe(true);
    expect(result.get_variable('__return__')).toBe(21);
  });

  it('captures thrown errors and tracebacks', () => {
    const result = execute_code('throw new Error("boom");');

    expect(result.success).toBe(false);
    expect(result.error).toBe('boom');
    expect(result.traceback).toContain('boom');
    expect(result.to_side_info_dict()).toMatchObject({
      Stdout: '',
      Stderr: '',
      Error: 'boom',
    });
  });

  it('supports subprocess execution for isolated snippets', () => {
    const result = execute_code('const answer = input + 1; console.error("warn");', {
      mode: ExecutionMode.SUBPROCESS,
      global_vars: { input: 41 },
      capture_variables: ['answer'],
    });

    expect(result.success).toBe(true);
    expect(result.stderr).toBe('warn\n');
    expect(result.get_variable('answer')).toBe(42);
  });

  it('reports timeout failures', () => {
    const result = execute_code('while (true) {}', {
      timeout: 0.05,
      mode: ExecutionMode.SUBPROCESS,
    });

    expect(result.success).toBe(false);
    expect(result.error).toContain('Timeout');
  });

  it('normalizes hashes like upstream', () => {
    expect(get_code_hash('x = 1   \n', 12)).toBe(get_code_hash('x = 1\n', 12));
    expect(get_code_hash('x = 1', 12)).toHaveLength(12);
  });
});
