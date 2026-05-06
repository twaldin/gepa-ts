declare module 'node:fs' {
  export function unlinkSync(path: string): void;
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
  env: Record<string, string | undefined>;
  stdout: { write(chunk: string): boolean };
  stderr: { write(chunk: string): boolean };
  on(event: 'SIGINT' | 'SIGTERM', listener: () => void): void;
  exit(code?: number): never;
};

declare const console: {
  log: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
};
