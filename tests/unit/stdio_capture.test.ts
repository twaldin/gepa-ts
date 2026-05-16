import { describe, expect, it } from 'vitest';
import { StreamCaptureManager, ThreadLocalStreamCapture } from '../../src/utils/stdio_capture.js';

describe('ThreadLocalStreamCapture', () => {
  it('captures writes between start_capture and stop_capture', () => {
    const passthrough: string[] = [];
    const capture = new ThreadLocalStreamCapture((text) => {
      passthrough.push(text);
      return true;
    });

    expect(capture.write('outside')).toBe(true);
    capture.start_capture();
    expect(capture.write('hello')).toBe(true);
    expect(capture.write(' world')).toBe(true);
    expect(capture.stop_capture()).toBe('hello world');
    expect(capture.stop_capture()).toBe('');
    expect(passthrough).toEqual(['outside']);
  });

  it('rejects nested capture on the same stream', () => {
    const capture = new ThreadLocalStreamCapture();
    capture.start_capture();

    expect(() => capture.start_capture()).toThrow('start_capture() called while already capturing');
    expect(capture.stop_capture()).toBe('');
  });
});

describe('StreamCaptureManager', () => {
  it('patches stdout and stderr with reference-counted release', () => {
    const manager = new StreamCaptureManager();
    const original_stdout = process.stdout.write;
    const original_stderr = process.stderr.write;

    const [stdout_capture, stderr_capture] = manager.acquire();
    manager.acquire();
    try {
      stdout_capture.start_capture();
      stderr_capture.start_capture();
      process.stdout.write('out');
      process.stderr.write('err');

      expect(stdout_capture.stop_capture()).toBe('out');
      expect(stderr_capture.stop_capture()).toBe('err');
    } finally {
      manager.release();
      expect(process.stdout.write).not.toBe(original_stdout);
      manager.release();
    }

    expect(process.stdout.write).toBe(original_stdout);
    expect(process.stderr.write).toBe(original_stderr);
  });
});
