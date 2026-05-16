declare module 'node:fs' {
  interface ReadBuffer {
    toString(encoding?: string): string;
  }
  export function unlinkSync(path: string): void;
  export function mkdirSync(path: string, opts?: { recursive?: boolean }): void;
  export function writeFileSync(path: string, data: string | Uint8Array): void;
  export function readFileSync(path: string): ReadBuffer;
  export function readFileSync(path: string, encoding: 'utf8'): string;
  export function existsSync(path: string): boolean;
  export function mkdtempSync(prefix: string): string;
  export function rmSync(path: string, opts?: { recursive?: boolean; force?: boolean }): void;
}

declare module 'node:os' {
  export function tmpdir(): string;
}

declare module 'node:readline' {
  export function createInterface(opts: { input: unknown; crlfDelay?: number }): {
    on(event: 'line', listener: (line: string) => void): void;
    close(): void;
  };
}

declare module 'node:async_hooks' {
  export class AsyncLocalStorage<T> {
    run<R>(store: T, callback: () => R): R;
    getStore(): T | undefined;
    enterWith(store: T): void;
  }
}

declare module 'node:net' {
  export interface Socket {
    write(chunk: string): void;
    on(event: 'error', listener: () => void): void;
    destroy(): void;
  }

  export interface Server {
    listen(path: string, callback: () => void): void;
    close(callback: () => void): void;
  }

  export function createServer(listener: (socket: Socket) => void): Server;
}

declare const process: {
  argv: string[];
  execPath: string;
  env: Record<string, string | undefined>;
  stdout: { write(chunk: string): boolean };
  stderr: { write(chunk: string): boolean };
  on(event: 'SIGINT' | 'SIGTERM', listener: () => void): void;
  off(event: 'SIGINT' | 'SIGTERM', listener: () => void): void;
  exit(code?: number): never;
};

declare const console: {
  log: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
};
