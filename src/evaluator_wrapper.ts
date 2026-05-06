import { AsyncLocalStorage } from 'node:async_hooks';
import {
  STR_CANDIDATE_KEY,
  type Candidate,
  type Evaluator,
  type EvaluatorOptState,
  type SideInfo,
} from './types';
import { LogContext, runWithLogContext } from './log_context';

const stdioAls = new AsyncLocalStorage<{ stdout: string[]; stderr: string[] } | null>();
let stdioPatched = false;

function patchStdioOnce(): void {
  if (stdioPatched) return;
  stdioPatched = true;

  const rawStdoutWrite = process.stdout.write.bind(process.stdout) as typeof process.stdout.write;
  const rawStderrWrite = process.stderr.write.bind(process.stderr) as typeof process.stderr.write;

  process.stdout.write = ((chunk: string) => {
    const bucket = stdioAls.getStore();
    if (bucket !== null && bucket !== undefined) {
      bucket.stdout.push(String(chunk));
      return true;
    }
    return rawStdoutWrite(chunk);
  }) as typeof process.stdout.write;

  process.stderr.write = ((chunk: string) => {
    const bucket = stdioAls.getStore();
    if (bucket !== null && bucket !== undefined) {
      bucket.stderr.push(String(chunk));
      return true;
    }
    return rawStderrWrite(chunk);
  }) as typeof process.stderr.write;
}

function runWithStdioCapture<T>(bucket: { stdout: string[]; stderr: string[] } | null, fn: () => Promise<T>): Promise<T> {
  if (bucket === null) return fn();
  patchStdioOnce();
  return stdioAls.run(bucket, fn);
}

function mergeCaptured(sideInfo: SideInfo, log: string, stdout: string, stderr: string): SideInfo {
  const out: SideInfo = { ...sideInfo };
  const captured: Array<{ key: 'log' | 'stdout' | 'stderr'; value: string }> = [];
  if (log !== '') captured.push({ key: 'log', value: log });
  if (stdout !== '') captured.push({ key: 'stdout', value: stdout });
  if (stderr !== '') captured.push({ key: 'stderr', value: stderr });

  for (const { key, value } of captured) {
    if (key in out) {
      const prefixed = `_gepa_${key}`;
      console.warn(
        `Your evaluator returned side_info with key '${key}' that conflicts with GEPA's captured output key. The captured output will be stored under '${prefixed}' instead.`,
      );
      out[prefixed] = value;
    } else {
      out[key] = value;
    }
  }
  return out;
}

export class EvaluatorWrapper {
  private readonly evaluator: Evaluator;
  private readonly single_instance_mode: boolean;
  private readonly capture_stdio: boolean;
  private readonly str_candidate_mode: boolean;
  private readonly raise_on_exception: boolean;

  constructor(
    evaluator: Evaluator,
    single_instance_mode: boolean,
    capture_stdio: boolean = false,
    str_candidate_mode: boolean = false,
    raise_on_exception: boolean = true,
  ) {
    this.evaluator = evaluator;
    this.single_instance_mode = single_instance_mode;
    this.capture_stdio = capture_stdio;
    this.str_candidate_mode = str_candidate_mode;
    this.raise_on_exception = raise_on_exception;
  }

  async call(
    candidate: Candidate,
    example?: unknown,
    opt_state: EvaluatorOptState = { best_example_evals: [] },
    providedLogContext?: LogContext,
  ): Promise<[number, unknown, SideInfo]> {
    const eval_candidate: string | Candidate = this.str_candidate_mode ? (candidate[STR_CANDIDATE_KEY] ?? '') : candidate;
    const logContext = providedLogContext ?? new LogContext();
    const stdioBucket = this.capture_stdio ? { stdout: [] as string[], stderr: [] as string[] } : null;

    try {
      const result = await runWithLogContext(logContext, () =>
        runWithStdioCapture(stdioBucket, () =>
          this.single_instance_mode
            ? Promise.resolve(this.evaluator(eval_candidate, { opt_state }))
            : Promise.resolve(this.evaluator(eval_candidate, { example, opt_state })),
        ),
      );

      const sideInfo = typeof result === 'number' ? {} : (result[1] ?? {});
      const score = typeof result === 'number' ? result : result[0];
      return [
        score,
        undefined,
        mergeCaptured(sideInfo, logContext.drain(), stdioBucket?.stdout.join('') ?? '', stdioBucket?.stderr.join('') ?? ''),
      ];
    } catch (error) {
      const merged = mergeCaptured(
        { error: String(error) },
        logContext.drain(),
        stdioBucket?.stdout.join('') ?? '',
        stdioBucket?.stderr.join('') ?? '',
      );
      if (this.raise_on_exception) throw error;
      return [0.0, undefined, merged];
    }
  }
}
