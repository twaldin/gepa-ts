import type { ChatMessage, LanguageModel } from './types.js';

export interface LMCompletionRequest {
  model: string;
  messages: ChatMessage[];
  num_retries: number;
  drop_params: true;
  [key: string]: unknown;
}

export interface LMBatchCompletionRequest {
  model: string;
  messages_list: ChatMessage[][];
  max_workers: number;
  num_retries: number;
  drop_params: true;
  [key: string]: unknown;
}

export interface LMChoice {
  message?: { content?: string | null };
  text?: string | null;
  finish_reason?: string | null;
}

export interface LMCompletionResponse {
  choices?: LMChoice[];
  usage?: {
    prompt_tokens?: number | null;
    completion_tokens?: number | null;
  } | null;
  cost?: number | null;
}

export type LMCompletionResult = string | LMCompletionResponse;
export type LMCompletionHook = (request: LMCompletionRequest) => LMCompletionResult | Promise<LMCompletionResult>;
export type LMBatchCompletionHook = (
  request: LMBatchCompletionRequest,
) => LMCompletionResult[] | Promise<LMCompletionResult[]>;

export interface LMOptions {
  temperature?: number | null;
  max_tokens?: number | null;
  num_retries?: number;
  completion?: LMCompletionHook;
  batch_completion?: LMBatchCompletionHook;
  [key: string]: unknown;
}

export interface LM extends LanguageModel {
  readonly model: string;
  readonly num_retries: number;
  readonly completion_kwargs: Record<string, unknown>;
  readonly total_cost: number;
  readonly total_tokens_in: number;
  readonly total_tokens_out: number;
  batch_complete(messages_list: ChatMessage[][], max_workers?: number, kwargs?: Record<string, unknown>): Promise<string[]>;
}

export interface TrackingLM extends LanguageModel {
  readonly total_cost: number;
  readonly total_tokens_in: number;
  readonly total_tokens_out: number;
  readonly wrapped_lm: LanguageModel;
}

type TrackableLanguageModel = LanguageModel & {
  total_cost?: unknown;
};

function estimate_tokens(value: string | ChatMessage[]): number {
  const text = typeof value === 'string' ? value : JSON.stringify(value);
  return Math.max(1, Math.floor(text.length / 4));
}

function normalize_messages(prompt: string | ChatMessage[]): ChatMessage[] {
  return typeof prompt === 'string' ? [{ role: 'user', content: prompt }] : prompt;
}

function token_count_from_response(result: LMCompletionResult, fallback_text: string): {
  input: number;
  output: number;
  cost: number;
} {
  if (typeof result === 'string') {
    return { input: 0, output: estimate_tokens(result), cost: 0 };
  }
  const usage = result.usage ?? null;
  const prompt_tokens = typeof usage?.prompt_tokens === 'number' ? usage.prompt_tokens : 0;
  const completion_tokens = typeof usage?.completion_tokens === 'number'
    ? usage.completion_tokens
    : estimate_tokens(fallback_text);
  const cost = typeof result.cost === 'number' ? result.cost : 0;
  return { input: prompt_tokens, output: completion_tokens, cost };
}

function text_from_completion(result: LMCompletionResult): string {
  if (typeof result === 'string') {
    return result;
  }
  const choice = result.choices?.[0];
  const content = choice?.message?.content ?? choice?.text;
  if (typeof content !== 'string') {
    throw new Error('LM completion response did not contain choices[0].message.content.');
  }
  return content;
}

function create_lm(model: string, options: LMOptions = {}): LM {
  const {
    completion,
    batch_completion,
    num_retries = 3,
    temperature,
    max_tokens,
    ...rest
  } = options;
  const completion_kwargs: Record<string, unknown> = {
    ...(temperature !== undefined && temperature !== null ? { temperature } : {}),
    ...(max_tokens !== undefined && max_tokens !== null ? { max_tokens } : {}),
    ...rest,
  };
  let total_cost_value = 0;
  let total_tokens_in_value = 0;
  let total_tokens_out_value = 0;

  const add_usage = (result: LMCompletionResult, text: string, prompt: string | ChatMessage[]): void => {
    const usage = token_count_from_response(result, text);
    total_cost_value += usage.cost;
    total_tokens_in_value += usage.input > 0 ? usage.input : estimate_tokens(prompt);
    total_tokens_out_value += usage.output;
  };

  const callable = Object.assign(
    async (prompt: string | ChatMessage[]) => {
      if (completion === undefined) {
        throw new Error(
          'LM requires an injected completion hook in the zero-dependency TypeScript build. ' +
            'Pass { completion } to new LM(...) or provide your own LanguageModel function.',
        );
      }
      const result = await completion({
        model,
        messages: normalize_messages(prompt),
        num_retries,
        drop_params: true,
        ...completion_kwargs,
      });
      const text = text_from_completion(result);
      add_usage(result, text, prompt);
      return text;
    },
    {
      model,
      num_retries,
      completion_kwargs,
      async batch_complete(
        messages_list: ChatMessage[][],
        max_workers = 10,
        kwargs: Record<string, unknown> = {},
      ): Promise<string[]> {
        if (batch_completion === undefined) {
          throw new Error(
            'LM.batch_complete requires an injected batch_completion hook in the zero-dependency TypeScript build.',
          );
        }
        const results = await batch_completion({
          model,
          messages_list,
          max_workers,
          num_retries,
          drop_params: true,
          ...completion_kwargs,
          ...kwargs,
        });
        return results.map((result, idx) => {
          const text = text_from_completion(result);
          add_usage(result, text, messages_list[idx] ?? []);
          return text.trim();
        });
      },
      toString() {
        const params = [`model='${model}'`];
        for (const [key, value] of Object.entries(completion_kwargs)) {
          params.push(`${key}=${JSON.stringify(value)}`);
        }
        return `LM(${params.join(', ')})`;
      },
    },
  ) as LM;

  Object.defineProperties(callable, {
    total_cost: { get: () => total_cost_value },
    total_tokens_in: { get: () => total_tokens_in_value },
    total_tokens_out: { get: () => total_tokens_out_value },
  });

  return callable;
}

function LMConstructor(this: LM | undefined, model: string, options: LMOptions = {}): LM {
  return create_lm(model, options);
}

export const LM = LMConstructor as {
  new (model: string, options?: LMOptions): LM;
  (model: string, options?: LMOptions): LM;
};

export function make_litellm_lm(model_name: string, options: LMOptions = {}): LM {
  return new LM(model_name, options);
}

function create_tracking_lm(fn: LanguageModel): TrackingLM {
  let total_tokens_in_value = 0;
  let total_tokens_out_value = 0;
  const callable = Object.assign(
    async (prompt: string | ChatMessage[]) => {
      total_tokens_in_value += estimate_tokens(prompt);
      const result = await fn(prompt);
      total_tokens_out_value += estimate_tokens(result);
      return result;
    },
    {
      wrapped_lm: fn,
    },
  ) as TrackingLM;
  Object.defineProperties(callable, {
    total_cost: { get: () => 0 },
    total_tokens_in: { get: () => total_tokens_in_value },
    total_tokens_out: { get: () => total_tokens_out_value },
  });
  return callable;
}

function TrackingLMConstructor(this: TrackingLM | undefined, fn: LanguageModel): TrackingLM {
  return create_tracking_lm(fn);
}

export const TrackingLM = TrackingLMConstructor as {
  new (fn: LanguageModel): TrackingLM;
  (fn: LanguageModel): TrackingLM;
};

export function ensure_tracking_lm<T extends LanguageModel>(lm: T): T | TrackingLM {
  const maybe_trackable: TrackableLanguageModel = lm;
  return typeof maybe_trackable.total_cost === 'number' ? lm : new TrackingLM(lm);
}
