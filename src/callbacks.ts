import type { GEPACallback } from "./types";

declare const console: { warn: (...args: unknown[]) => void };

function is_callback_method(value: unknown): value is (event: unknown) => void {
  return typeof value === "function";
}

export function notify_callbacks<E>(callbacks: GEPACallback[] | undefined, method: string, event: E): void {
  if (callbacks === undefined) {
    return;
  }

  for (const callback of callbacks) {
    const method_candidate = (callback as Record<string, unknown>)[method];
    if (is_callback_method(method_candidate)) {
      try {
        method_candidate(event);
      } catch (error) {
        console.warn(`Callback failed on ${method}:`, error);
      }
    }
  }
}
