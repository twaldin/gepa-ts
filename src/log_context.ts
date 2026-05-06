import { AsyncLocalStorage } from 'node:async_hooks';

export class LogContext {
  private chunks: string[] = [];

  write(text: string): void {
    this.chunks.push(text);
  }

  drain(): string {
    const out = this.chunks.join('');
    this.chunks = [];
    return out;
  }
}

const logContextAls = new AsyncLocalStorage<LogContext | null>();
let warnedOutside = false;

export function getLogContext(): LogContext | null {
  return logContextAls.getStore() ?? null;
}

export function getLogContextOrThrow(): LogContext {
  const ctx = getLogContext();
  if (ctx === null) {
    throw new Error(
      'No active log context. get_log_context() must be called inside an evaluator passed to optimize_anything().',
    );
  }
  return ctx;
}

export function setLogContext(ctx: LogContext | null): void {
  logContextAls.enterWith(ctx);
}

export function runWithLogContext<T>(ctx: LogContext | null, fn: () => T): T {
  return logContextAls.run(ctx, fn);
}

export function oa_log(...args: unknown[]): void;
export function oa_log(...args: [...unknown[], { sep?: string; end?: string }]): void;
export function oa_log(...args: unknown[]): void {
  let sep = ' ';
  let end = '\n';
  let parts = args;

  const maybeLast = args[args.length - 1];
  if (
    typeof maybeLast === 'object' &&
    maybeLast !== null &&
    (Object.prototype.hasOwnProperty.call(maybeLast, 'sep') || Object.prototype.hasOwnProperty.call(maybeLast, 'end'))
  ) {
    const opts = maybeLast as { sep?: unknown; end?: unknown };
    if (typeof opts.sep === 'string') sep = opts.sep;
    if (typeof opts.end === 'string') end = opts.end;
    parts = args.slice(0, -1);
  }

  const ctx = getLogContext();
  if (ctx === null) {
    if (!warnedOutside) {
      warnedOutside = true;
      console.warn(
        'oa.log() called outside of an evaluator function. Output will be discarded. Only call oa.log() inside your evaluator, or propagate the log context to child threads via oa.get_log_context() / oa.set_log_context().',
      );
    }
    return;
  }

  ctx.write(parts.map((v) => String(v)).join(sep) + end);
}
