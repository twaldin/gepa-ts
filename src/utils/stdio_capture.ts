export class ThreadLocalStreamCapture {
  private buffer: string[] | null = null;

  constructor(private readonly original_write: (text: string) => boolean = (text) => process.stdout.write(text)) {}

  write(text: string): boolean {
    if (this.buffer !== null) {
      this.buffer.push(text);
      return true;
    }
    return this.original_write(text);
  }

  flush(): void {
    return;
  }

  isatty(): boolean {
    return false;
  }

  writable(): boolean {
    return true;
  }

  readable(): boolean {
    return false;
  }

  start_capture(): void {
    if (this.buffer !== null) {
      throw new Error('start_capture() called while already capturing on this stream.');
    }
    this.buffer = [];
  }

  stop_capture(): string {
    if (this.buffer === null) {
      return '';
    }
    const text = this.buffer.join('');
    this.buffer = null;
    return text;
  }
}

export class StreamCaptureManager {
  private refcount = 0;
  private original_stdout_write: ((text: string) => boolean) | null = null;
  private original_stderr_write: ((text: string) => boolean) | null = null;
  private stdout_capturer: ThreadLocalStreamCapture | null = null;
  private stderr_capturer: ThreadLocalStreamCapture | null = null;

  acquire(): [ThreadLocalStreamCapture, ThreadLocalStreamCapture] {
    if (this.refcount === 0) {
      const stdout_write = process.stdout.write;
      const stderr_write = process.stderr.write;
      this.original_stdout_write = stdout_write;
      this.original_stderr_write = stderr_write;
      this.stdout_capturer = new ThreadLocalStreamCapture((text) => stdout_write.call(process.stdout, text));
      this.stderr_capturer = new ThreadLocalStreamCapture((text) => stderr_write.call(process.stderr, text));
      process.stdout.write = ((chunk: string) => this.stdout_capturer?.write(String(chunk)) ?? true) as typeof process.stdout.write;
      process.stderr.write = ((chunk: string) => this.stderr_capturer?.write(String(chunk)) ?? true) as typeof process.stderr.write;
    }
    this.refcount += 1;
    if (this.stdout_capturer === null || this.stderr_capturer === null) {
      throw new Error('Stream capture manager failed to initialize.');
    }
    return [this.stdout_capturer, this.stderr_capturer];
  }

  release(): void {
    this.refcount -= 1;
    if (this.refcount > 0) {
      return;
    }
    if (this.original_stdout_write !== null) {
      process.stdout.write = this.original_stdout_write as typeof process.stdout.write;
    }
    if (this.original_stderr_write !== null) {
      process.stderr.write = this.original_stderr_write as typeof process.stderr.write;
    }
    this.refcount = 0;
    this.original_stdout_write = null;
    this.original_stderr_write = null;
    this.stdout_capturer = null;
    this.stderr_capturer = null;
  }
}

export const stream_manager = new StreamCaptureManager();
