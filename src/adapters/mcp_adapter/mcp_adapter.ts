import type { Candidate, EvaluationBatch, GEPAAdapter } from '../../types.js';
import { create_mcp_client } from './mcp_client.js';

export type MCPDataInst = {
  user_query: string;
  tool_arguments: Record<string, unknown>;
  reference_answer: string | null;
  additional_context: Record<string, string>;
};

export type MCPTrajectory = {
  user_query: string;
  tool_names: string[];
  selected_tool: string | null;
  tool_called: boolean;
  tool_arguments: Record<string, unknown> | null;
  tool_response: string | null;
  tool_description_used: string;
  system_prompt_used: string;
  model_first_pass_output: string;
  model_final_output: string;
  score: number;
};

export type MCPOutput = {
  final_answer: string;
  tool_called: boolean;
  selected_tool: string | null;
  tool_response: string | null;
};

export type MCPChatMessage = {
  role: string;
  content: string;
};

export type MCPToolDefinition = {
  name?: string;
  description?: string;
  inputSchema?: {
    properties?: Record<string, { type?: string }>;
    [key: string]: unknown;
  };
  [key: string]: unknown;
};

export type MCPClient = {
  start(): Promise<void>;
  initialize(): Promise<Record<string, unknown>>;
  list_tools(): Promise<MCPToolDefinition[]>;
  call_tool(tool_name: string, tool_arguments: Record<string, unknown>): Promise<unknown>;
  close(): Promise<void>;
};

export type MCPClientFactory = () => MCPClient;
export type MCPTaskModel = (messages: MCPChatMessage[]) => string | Promise<string>;
export type MCPMetricFn = (item: MCPDataInst, output: string) => number;

export type MCPAdapterConfig = {
  tool_names: string | string[];
  task_model: MCPTaskModel | string;
  metric_fn: MCPMetricFn;
  server_params?: unknown;
  remote_url?: string | null;
  remote_transport?: 'sse' | 'streamable_http' | string;
  remote_headers?: Record<string, string> | null;
  remote_timeout?: number;
  base_system_prompt?: string;
  enable_two_pass?: boolean;
  failure_score?: number;
  client_factory?: MCPClientFactory | null;
};

type StdioServerParamsLike = {
  command: string;
  args?: string[];
};

type FirstPassResult = {
  output: string;
  tool_called: boolean;
  selected_tool: string | null;
  tool_arguments: Record<string, unknown> | null;
  tool_response: string | null;
};

function normalize_tool_names(tool_names: string | string[]): string[] {
  return Array.isArray(tool_names) ? tool_names : [tool_names];
}

function is_record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function is_stdio_server_params(value: unknown): value is StdioServerParamsLike {
  return is_record(value) && typeof value.command === 'string' && (value.args === undefined || Array.isArray(value.args));
}

function content_to_text(content_item: unknown): string | null {
  if (!is_record(content_item)) return null;
  if (content_item.type === 'text' && typeof content_item.text === 'string') {
    return content_item.text;
  }
  if ('text' in content_item && typeof content_item.text === 'string') {
    return content_item.text;
  }
  if (content_item.type === 'image') {
    const mime_type = typeof content_item.mimeType === 'string' ? content_item.mimeType : 'unknown';
    const data = content_item.data;
    const data_len = typeof data === 'string' || Array.isArray(data) ? data.length : 0;
    return `[Image: ${mime_type}, ${data_len} bytes]`;
  }
  return null;
}

function format_score(score: number): string {
  return score.toFixed(2);
}

export class MCPAdapter implements GEPAAdapter<MCPDataInst, MCPTrajectory, MCPOutput> {
  readonly server_params: unknown;
  readonly remote_url: string | null;
  readonly remote_transport: string;
  readonly remote_headers: Record<string, string>;
  readonly remote_timeout: number;
  readonly tool_names: string[];
  readonly base_system_prompt: string;
  readonly enable_two_pass: boolean;
  readonly failure_score: number;
  readonly metric_fn: MCPMetricFn;
  readonly task_model: MCPTaskModel | string;
  private readonly client_factory: MCPClientFactory | null;

  constructor({
    tool_names,
    task_model,
    metric_fn,
    server_params = null,
    remote_url = null,
    remote_transport = 'sse',
    remote_headers = null,
    remote_timeout = 30,
    base_system_prompt = 'You are a helpful assistant with access to tools.',
    enable_two_pass = true,
    failure_score = 0.0,
    client_factory = null,
  }: MCPAdapterConfig) {
    this.server_params = server_params;
    this.remote_url = remote_url;
    this.remote_transport = remote_transport;
    this.remote_headers = remote_headers ?? {};
    this.remote_timeout = remote_timeout;
    this.tool_names = normalize_tool_names(tool_names);
    this.base_system_prompt = base_system_prompt;
    this.enable_two_pass = enable_two_pass;
    this.failure_score = failure_score;
    this.metric_fn = metric_fn;
    this.task_model = task_model;
    this.client_factory = client_factory;
  }

  async evaluate(
    batch: MCPDataInst[],
    candidate: Candidate,
    capture_traces: boolean = false,
  ): Promise<EvaluationBatch<MCPTrajectory, MCPOutput>> {
    const outputs: MCPOutput[] = [];
    const scores: number[] = [];
    const trajectories: MCPTrajectory[] | undefined = capture_traces ? [] : undefined;

    let client: MCPClient | null = null;
    try {
      client = this.client_factory === null
        ? create_mcp_client({
            server_params: is_stdio_server_params(this.server_params) ? this.server_params : null,
            remote_url: this.remote_url,
            remote_transport: this.remote_transport,
            remote_headers: this.remote_headers,
            remote_timeout: this.remote_timeout,
          })
        : this.client_factory();
      await client.start();
      await client.initialize();
      const tools_list = await client.list_tools();
      const available_tools = tools_list.filter((tool) => typeof tool.name === 'string' && this.tool_names.includes(tool.name));
      if (available_tools.length === 0) {
        const available_names = tools_list.map((tool) => tool.name);
        throw new Error(`Tools ${JSON.stringify(this.tool_names)} not found. Available: ${JSON.stringify(available_names)}`);
      }

      const system_prompt = this._build_system_prompt(candidate, available_tools);
      for (const item of batch) {
        try {
          const first_pass = await this._first_pass(client, item, system_prompt);
          const final_output =
            this.enable_two_pass && first_pass.tool_called
              ? await this._second_pass(item, system_prompt, first_pass.tool_response)
              : first_pass.output;
          const score = this.metric_fn(item, final_output);
          outputs.push({
            final_answer: final_output,
            tool_called: first_pass.tool_called,
            selected_tool: first_pass.selected_tool,
            tool_response: first_pass.tool_response,
          });
          scores.push(score);
          if (trajectories !== undefined) {
            trajectories.push({
              user_query: item.user_query,
              tool_names: this.tool_names,
              selected_tool: first_pass.selected_tool,
              tool_called: first_pass.tool_called,
              tool_arguments: first_pass.tool_arguments,
              tool_response: first_pass.tool_response,
              tool_description_used: candidate.tool_description ?? '',
              system_prompt_used: system_prompt,
              model_first_pass_output: first_pass.output,
              model_final_output: final_output,
              score,
            });
          }
        } catch (error) {
          this.push_failure(item, outputs, scores, trajectories, candidate.tool_description ?? '', system_prompt, `ERROR: ${String(error)}`);
        }
      }
    } catch (error) {
      for (const item of batch) {
        this.push_failure(item, outputs, scores, trajectories, '', '', `SESSION ERROR: ${String(error)}`);
      }
    } finally {
      if (client !== null) {
        await client.close();
      }
    }

    return { outputs, scores, ...(trajectories !== undefined ? { trajectories } : {}), num_metric_calls: batch.length };
  }

  private push_failure(
    item: MCPDataInst,
    outputs: MCPOutput[],
    scores: number[],
    trajectories: MCPTrajectory[] | undefined,
    tool_description_used: string,
    system_prompt_used: string,
    error_text: string,
  ): void {
    outputs.push({ final_answer: '', tool_called: false, selected_tool: null, tool_response: null });
    scores.push(this.failure_score);
    if (trajectories !== undefined) {
      trajectories.push({
        user_query: item.user_query,
        tool_names: this.tool_names,
        selected_tool: null,
        tool_called: false,
        tool_arguments: null,
        tool_response: null,
        tool_description_used,
        system_prompt_used,
        model_first_pass_output: error_text,
        model_final_output: '',
        score: this.failure_score,
      });
    }
  }

  private async call_model(messages: MCPChatMessage[]): Promise<string> {
    if (typeof this.task_model === 'string') {
      throw new Error('String model execution is not bundled in the zero-dependency TypeScript adapter; pass a callable task_model.');
    }
    return this.task_model(messages);
  }

  private async _first_pass(client: MCPClient, item: MCPDataInst, system_prompt: string): Promise<FirstPassResult> {
    const model_output = (await this.call_model([
      { role: 'system', content: system_prompt },
      { role: 'user', content: item.user_query },
    ])).trim();

    let parsed: unknown;
    try {
      parsed = JSON.parse(model_output);
    } catch {
      return { output: model_output, tool_called: false, selected_tool: null, tool_arguments: null, tool_response: null };
    }
    if (!is_record(parsed) || parsed.action !== 'call_tool' || typeof parsed.tool !== 'string') {
      return { output: model_output, tool_called: false, selected_tool: null, tool_arguments: null, tool_response: null };
    }
    if (!this.tool_names.includes(parsed.tool)) {
      return { output: model_output, tool_called: false, selected_tool: null, tool_arguments: null, tool_response: null };
    }
    const tool_arguments = is_record(parsed.arguments) ? parsed.arguments : {};
    const tool_result = await client.call_tool(parsed.tool, tool_arguments);
    return {
      output: model_output,
      tool_called: true,
      selected_tool: parsed.tool,
      tool_arguments,
      tool_response: this._extract_tool_response(tool_result),
    };
  }

  private async _second_pass(item: MCPDataInst, system_prompt: string, tool_response: string | null): Promise<string> {
    return (await this.call_model([
      { role: 'system', content: system_prompt },
      { role: 'user', content: item.user_query },
      { role: 'assistant', content: `I'll use the tool to help answer this. Tool response: ${tool_response}` },
      { role: 'user', content: 'Based on the tool response, please provide your final answer.' },
    ])).trim();
  }

  _build_system_prompt(candidate: Candidate, available_tools: MCPToolDefinition[]): string {
    const custom_system_prompt = candidate.system_prompt ?? this.base_system_prompt;
    let tools_section = 'You have access to the following tools:\n\n';
    for (const tool of available_tools) {
      const tool_name = tool.name ?? '';
      const optimized_desc = candidate[`tool_description_${tool_name}`] ?? candidate.tool_description;
      const tool_description = optimized_desc ?? tool.description ?? '';
      const input_schema = tool.inputSchema ?? {};
      const properties = input_schema.properties ?? {};
      const example_args: Record<string, unknown> = {};
      for (const [param_name, param_info] of Object.entries(properties)) {
        if (param_info.type === 'string') example_args[param_name] = 'example_value';
        else if (param_info.type === 'number') example_args[param_name] = 123;
        else if (param_info.type === 'boolean') example_args[param_name] = true;
        else example_args[param_name] = 'value';
      }
      if (Object.keys(example_args).length === 0) {
        example_args.param = 'value';
      }
      tools_section +=
        `Tool: ${tool_name}\n` +
        `Description: ${tool_description}\n` +
        `Input Schema: ${JSON.stringify(input_schema, null, 2)}\n` +
        `Example usage: ${JSON.stringify({ action: 'call_tool', tool: tool_name, arguments: example_args })}\n\n`;
    }
    const usage_instructions =
      '\nWhen you need to use a tool, respond ONLY with JSON:\n' +
      '{"action": "call_tool", "tool": "tool_name", "arguments": {"param": "value"}}\n\n' +
      'When you can answer directly, respond ONLY with JSON:\n' +
      '{"action": "answer", "text": "your answer"}\n\n' +
      `Choose the most appropriate tool for the task. Available tools: ${JSON.stringify(available_tools.map((tool) => tool.name))}\n\n` +
      'Always respond with valid JSON. No other text.\n';
    return `${custom_system_prompt}\n${tools_section}${usage_instructions}`;
  }

  _extract_tool_response(result: unknown): string {
    if (!is_record(result)) return String(result);
    if (result.isError === true && Array.isArray(result.content)) {
      const error_texts = result.content.map(content_to_text).filter((text): text is string => text !== null);
      return `ERROR: ${error_texts.length > 0 ? error_texts.join('\n') : 'Tool execution failed'}`;
    }
    if ('structuredContent' in result && result.structuredContent !== null && result.structuredContent !== undefined) {
      return JSON.stringify(result.structuredContent, null, 2);
    }
    if (Array.isArray(result.content)) {
      const texts = result.content.map(content_to_text).filter((text): text is string => text !== null);
      return texts.join('\n');
    }
    return String(result);
  }

  make_reflective_dataset(
    _candidate: Candidate,
    eval_batch: EvaluationBatch<MCPTrajectory, MCPOutput>,
    components_to_update: string[],
  ): Record<string, Array<Record<string, unknown>>> {
    const reflective_data: Record<string, Array<Record<string, unknown>>> = {};
    for (const component of components_to_update) {
      reflective_data[component] = [];
      for (const [idx, traj] of (eval_batch.trajectories ?? []).entries()) {
        const score = eval_batch.scores[idx] ?? this.failure_score;
        if (component === 'tool_description') {
          reflective_data[component]!.push({
            Inputs: {
              user_query: traj.user_query,
              tool_description: traj.tool_description_used,
            },
            'Generated Outputs': {
              tool_called: traj.tool_called,
              selected_tool: traj.selected_tool,
              tool_arguments: traj.tool_arguments,
              final_answer: traj.model_final_output,
            },
            Feedback: this._generate_tool_feedback(traj, score),
          });
        } else if (component === 'system_prompt') {
          reflective_data[component]!.push({
            Inputs: {
              user_query: traj.user_query,
              system_prompt: traj.system_prompt_used,
            },
            'Generated Outputs': traj.model_final_output,
            Feedback: this._generate_system_prompt_feedback(traj, score),
          });
        }
      }
    }
    return reflective_data;
  }

  private _generate_tool_feedback(traj: MCPTrajectory, score: number): string {
    if (score > 0.5) {
      if (traj.tool_called) {
        return `Good! The tool '${traj.selected_tool}' was used appropriately. Score: ${format_score(score)}`;
      }
      return `Good! No tool needed, direct answer was correct. Score: ${format_score(score)}`;
    }
    const feedback_parts = [`Incorrect response (score: ${format_score(score)}).`];
    if (!traj.tool_called) {
      feedback_parts.push('Tool was not called. Consider if a tool would help.');
    } else {
      feedback_parts.push(`Tool '${traj.selected_tool}' was called with ${JSON.stringify(traj.tool_arguments)}, but answer was incorrect.`);
      if (traj.tool_names.length > 1) {
        feedback_parts.push(`Consider a different tool from ${JSON.stringify(traj.tool_names)} or clearer description.`);
      } else {
        feedback_parts.push('Tool description may need improvement.');
      }
    }
    return feedback_parts.join(' ');
  }

  private _generate_system_prompt_feedback(traj: MCPTrajectory, score: number): string {
    if (score > 0.5) {
      return `System prompt provided good guidance. Score: ${format_score(score)}`;
    }
    return (
      `System prompt may need improvement (score: ${format_score(score)}). ` +
      `Model ${traj.tool_called ? 'called' : 'did not call'} tool, but answer was incorrect.`
    );
  }
}
