declare module 'node:fs' {
  export function unlinkSync(path: string): void;
}

declare module 'node:readline' {
  export function createInterface(opts: { input: unknown; crlfDelay?: number }): {
    on(event: 'line', listener: (line: string) => void): void;
    close(): void;
  };
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
  stderr: { write(chunk: string): void };
  on(event: 'SIGINT' | 'SIGTERM', listener: () => void): void;
  exit(code?: number): never;
};
