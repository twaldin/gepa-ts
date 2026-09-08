import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { Script, createContext } from 'node:vm';

export enum ExecutionMode {
  IN_PROCESS = 'in_process',
  SUBPROCESS = 'subprocess',
}

export class TimeLimitError extends Error {
  constructor(message = 'Code execution timed out') {
    super(message);
    this.name = 'TimeLimitError';
  }
}

export interface CodeExecutionResultInit {
  success: boolean;
  stdout?: string;
  stderr?: string;
  error?: string;
  traceback?: string;
  variables?: Record<string, unknown>;
  execution_time?: number;
  code_hash?: string;
}

export class CodeExecutionResult {
  success: boolean;
  stdout: string;
  stderr: string;
  error: string;
  traceback: string;
  variables: Record<string, unknown>;
  execution_time: number;
  code_hash: string;

  constructor(init: CodeExecutionResultInit) {
    this.success = init.success;
    this.stdout = init.stdout ?? '';
    this.stderr = init.stderr ?? '';
    this.error = init.error ?? '';
    this.traceback = init.traceback ?? '';
    this.variables = init.variables ?? {};
    this.execution_time = init.execution_time ?? 0;
    this.code_hash = init.code_hash ?? '';
  }

  get_variable(name: string, default_value: unknown = null): unknown {
    return Object.prototype.hasOwnProperty.call(this.variables, name)
      ? this.variables[name]
      : default_value;
  }

  to_side_info_dict(): Record<string, string> {
    const info: Record<string, string> = {
      Stdout: this.stdout,
      Stderr: this.stderr,
    };
    if (this.error) {
      info.Error = this.error;
    }
    if (this.traceback) {
      info.Traceback = this.traceback;
    }
    return info;
  }
}

export interface ExecuteCodeOptions {
  timeout?: number;
  mode?: ExecutionMode | `${ExecutionMode}`;
  global_vars?: Record<string, unknown> | null;
  entry_point?: string | null;
  entry_point_args?: unknown[];
  entry_point_kwargs?: Record<string, unknown> | null;
  capture_variables?: string[] | null;
  seed?: number | null;
  kill_child_processes?: boolean;
}

interface RunOutput {
  success: boolean;
  stdout: string;
  stderr: string;
  error: string;
  traceback: string;
  variables: Record<string, unknown>;
}

function compute_code_hash(code: string): string {
  const normalized = code.trim().split('\n').map((line) => line.trimEnd()).join('\n');
  return createHash('sha256').update(normalized).digest('hex');
}

function serialize_for_js(value: unknown): string {
  return JSON.stringify(value, (_key, nested) => {
    if (typeof nested === 'bigint') {
      return nested.toString();
    }
    if (typeof nested === 'function' || typeof nested === 'symbol') {
      return undefined;
    }
    return nested;
  }) ?? 'null';
}

function format_console(args: unknown[]): string {
  return args.map((arg) => {
    if (typeof arg === 'string') {
      return arg;
    }
    if (arg instanceof Error) {
      return arg.stack ?? arg.message;
    }
    try {
      return JSON.stringify(arg);
    } catch {
      return String(arg);
    }
  }).join(' ');
}

function normalize_error_message(message: string): string {
  return message.startsWith('Error: ') ? message.slice('Error: '.length) : message;
}

function build_capture_expression(capture_variables: string[] | null | undefined): string {
  if (capture_variables != null) {
    const entries = capture_variables.map((name) => {
      const encoded = JSON.stringify(name);
      return `${encoded}: (typeof ${name} !== "undefined" ? ${name} : globalThis[${encoded}])`;
    });
    return `({${entries.join(',')}})`;
  }
  return `Object.fromEntries(Object.entries(globalThis).filter(([key, value]) =>
    !key.startsWith("__") &&
    key !== "console" &&
    key !== "globalThis" &&
    typeof value !== "function"
  ))`;
}

function entry_point_call_expression(entry_point: string, entry_point_kwargs: Record<string, unknown> | null | undefined): string {
  const encoded = JSON.stringify(entry_point);
  const has_kwargs = entry_point_kwargs != null && Object.keys(entry_point_kwargs).length > 0;
  return has_kwargs
    ? `globalThis.__return__ = globalThis[${encoded}](...globalThis.__entry_point_args__, globalThis.__entry_point_kwargs__);`
    : `globalThis.__return__ = globalThis[${encoded}](...globalThis.__entry_point_args__);`;
}

function make_wrapped_code(code: string, options: Required<Pick<ExecuteCodeOptions, 'entry_point_args'>> & ExecuteCodeOptions): string {
  const capture_expression = build_capture_expression(options.capture_variables);
  const entry_call = options.entry_point != null
    ? `
if (typeof ${JSON.stringify(options.entry_point)} === "string" && typeof globalThis[${JSON.stringify(options.entry_point)}] === "function") {
  ${entry_point_call_expression(options.entry_point, options.entry_point_kwargs)}
}`
    : '';

  return `
${code}
${entry_call}
globalThis.__gepa_variables__ = ${capture_expression};
if (typeof globalThis.__return__ !== "undefined") {
  globalThis.__gepa_variables__.__return__ = globalThis.__return__;
}
`;
}

function make_console(stdout: string[], stderr: string[]): Console {
  const capture = (target: string[]) => (...args: unknown[]) => {
    target.push(`${format_console(args)}\n`);
  };
  return {
    log: capture(stdout),
    info: capture(stdout),
    warn: capture(stderr),
    error: capture(stderr),
    debug: capture(stdout),
  } as Console;
}

function execute_in_process(code: string, timeout: number, options: ExecuteCodeOptions, code_hash: string): CodeExecutionResult {
  const start = performance.now();
  const stdout: string[] = [];
  const stderr: string[] = [];
  const context_seed = {
    ...(options.global_vars ?? {}),
    console: make_console(stdout, stderr),
    __entry_point_args__: options.entry_point_args ?? [],
    __entry_point_kwargs__: options.entry_point_kwargs ?? {},
  };
  const context = createContext(context_seed);
  const wrapped_code = make_wrapped_code(code, { ...options, entry_point_args: options.entry_point_args ?? [] });

  try {
    const script = new Script(wrapped_code);
    script.runInContext(context, timeout > 0 ? { timeout: Math.max(1, Math.ceil(timeout * 1000)) } : undefined);
    const variables = context.__gepa_variables__;
    const elapsed = (performance.now() - start) / 1000;
    return new CodeExecutionResult({
      success: true,
      stdout: stdout.join(''),
      stderr: stderr.join(''),
      variables: variables != null && typeof variables === 'object' ? variables as Record<string, unknown> : {},
      execution_time: elapsed,
      code_hash,
    });
  } catch (error) {
    const elapsed = (performance.now() - start) / 1000;
    const message = normalize_error_message(error instanceof Error ? error.message : String(error));
    const stack = error instanceof Error ? error.stack ?? message : message;
    const is_timeout = message.includes('Script execution timed out');
    return new CodeExecutionResult({
      success: false,
      stdout: stdout.join(''),
      stderr: stderr.join(''),
      error: is_timeout ? `Timeout: execution exceeded ${timeout} seconds` : message,
      traceback: stack,
      execution_time: elapsed,
      code_hash,
    });
  }
}

function subprocess_wrapper(): string {
  return `
const vm = require("node:vm");
let payload = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { payload += chunk; });
process.stdin.on("end", () => {
  const input = JSON.parse(payload);
  const stdout = [];
  const stderr = [];
  const format = (args) => args.map((arg) => {
    if (typeof arg === "string") return arg;
    if (arg && typeof arg === "object" && typeof arg.stack === "string") return arg.stack;
    try { return JSON.stringify(arg); } catch { return String(arg); }
  }).join(" ");
  const capture = (target) => (...args) => target.push(format(args) + "\\n");
  const context = vm.createContext({
    ...(input.global_vars || {}),
    console: {
      log: capture(stdout),
      info: capture(stdout),
      warn: capture(stderr),
      error: capture(stderr),
      debug: capture(stdout),
    },
    __entry_point_args__: input.entry_point_args || [],
    __entry_point_kwargs__: input.entry_point_kwargs || {},
  });
  const captureExpression = input.capture_variables
    ? "({" + input.capture_variables.map((name) => JSON.stringify(name) + ": (typeof " + name + " !== \\"undefined\\" ? " + name + " : globalThis[" + JSON.stringify(name) + "])").join(",") + "})"
    : "Object.fromEntries(Object.entries(globalThis).filter(([key, value]) => !key.startsWith(\\"__\\") && key !== \\"console\\" && key !== \\"globalThis\\" && typeof value !== \\"function\\"))";
  const entry = input.entry_point
    ? "\\nif (typeof globalThis[" + JSON.stringify(input.entry_point) + "] === \\"function\\") { globalThis.__return__ = globalThis[" + JSON.stringify(input.entry_point) + "](...globalThis.__entry_point_args__" + (input.entry_point_kwargs && Object.keys(input.entry_point_kwargs).length > 0 ? ", globalThis.__entry_point_kwargs__" : "") + "); }\\n"
    : "";
  const wrapped = input.code + entry + "\\nglobalThis.__gepa_variables__ = " + captureExpression + ";\\nif (typeof globalThis.__return__ !== \\"undefined\\") { globalThis.__gepa_variables__.__return__ = globalThis.__return__; }\\n";
  try {
    new vm.Script(wrapped).runInContext(context, input.timeout > 0 ? { timeout: Math.max(1, Math.ceil(input.timeout * 1000)) } : undefined);
    process.stdout.write(JSON.stringify({ success: true, stdout: stdout.join(""), stderr: stderr.join(""), error: "", traceback: "", variables: context.__gepa_variables__ || {} }));
  } catch (error) {
    const message = error && typeof error.message === "string" ? error.message : String(error);
    const stack = error && typeof error.stack === "string" ? error.stack : message;
    process.stdout.write(JSON.stringify({ success: false, stdout: stdout.join(""), stderr: stderr.join(""), error: message.includes("Script execution timed out") ? "Timeout: execution exceeded " + input.timeout + " seconds" : message, traceback: stack, variables: {} }));
  }
});
`;
}

function execute_subprocess(code: string, timeout: number, options: ExecuteCodeOptions, code_hash: string): CodeExecutionResult {
  const start = performance.now();
  const payload = {
    code,
    timeout,
    global_vars: options.global_vars ?? {},
    entry_point: options.entry_point ?? null,
    entry_point_args: options.entry_point_args ?? [],
    entry_point_kwargs: options.entry_point_kwargs ?? {},
    capture_variables: options.capture_variables ?? null,
  };
  const run = spawnSync(process.execPath, ['-e', subprocess_wrapper()], {
    input: serialize_for_js(payload),
    encoding: 'utf8',
    timeout: timeout > 0 ? Math.max(1, Math.ceil(timeout * 1000) + 100) : undefined,
    maxBuffer: 10 * 1024 * 1024,
  });
  const elapsed = (performance.now() - start) / 1000;

  if (run.error != null) {
    const message = run.error.message.includes('ETIMEDOUT')
      ? `Timeout: execution exceeded ${timeout} seconds`
      : normalize_error_message(run.error.message);
    return new CodeExecutionResult({
      success: false,
      stdout: run.stdout ?? '',
      stderr: run.stderr ?? '',
      error: message,
      traceback: run.stderr ?? message,
      execution_time: elapsed,
      code_hash,
    });
  }

  try {
    const parsed = JSON.parse(run.stdout) as RunOutput;
    return new CodeExecutionResult({
      success: parsed.success,
      stdout: parsed.stdout,
      stderr: parsed.stderr,
      error: parsed.error,
      traceback: parsed.traceback,
      variables: parsed.variables,
      execution_time: elapsed,
      code_hash,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return new CodeExecutionResult({
      success: false,
      stdout: run.stdout ?? '',
      stderr: run.stderr ?? '',
      error: `Failed to load results: ${message}`,
      traceback: run.stderr ?? message,
      execution_time: elapsed,
      code_hash,
    });
  }
}

export function execute_code(code: string, options: ExecuteCodeOptions = {}): CodeExecutionResult {
  const timeout = options.timeout ?? 30;
  const mode = options.mode ?? ExecutionMode.IN_PROCESS;
  const code_hash = compute_code_hash(code);
  return mode === ExecutionMode.SUBPROCESS
    ? execute_subprocess(code, timeout, options, code_hash)
    : execute_in_process(code, timeout, options, code_hash);
}

export function get_code_hash(code: string, length = 8): string {
  return compute_code_hash(code).slice(0, length);
}
