import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';

import type { MCPClient, MCPToolDefinition } from './mcp_adapter.js';

export type StdioServerParameters = {
  command: string;
  args?: string[];
};

type JsonRpcId = number;

type JsonRpcRequest = {
  jsonrpc: '2.0';
  method: string;
  id?: JsonRpcId;
  params?: Record<string, unknown>;
};

type JsonRpcResponse = {
  jsonrpc?: string;
  id?: JsonRpcId;
  result?: unknown;
  error?: unknown;
};

function is_record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function parse_json_rpc_response(line: string): JsonRpcResponse {
  const parsed: unknown = JSON.parse(line);
  if (!is_record(parsed)) {
    throw new Error(`Unexpected MCP response: ${line}`);
  }
  const response: JsonRpcResponse = {};
  if (typeof parsed.jsonrpc === 'string') response.jsonrpc = parsed.jsonrpc;
  if (typeof parsed.id === 'number') response.id = parsed.id;
  if ('result' in parsed) response.result = parsed.result;
  if ('error' in parsed) response.error = parsed.error;
  return response;
}

function normalize_result_dict(result: unknown): Record<string, unknown> {
  return is_record(result) ? result : {};
}

function normalize_tools(value: unknown): MCPToolDefinition[] {
  if (!Array.isArray(value)) return [];
  return value.filter(is_record) as MCPToolDefinition[];
}

async function read_json_response(response: Response): Promise<JsonRpcResponse> {
  if (!response.ok) {
    throw new Error(`MCP HTTP error: ${response.status} ${response.statusText}`);
  }
  const parsed: unknown = await response.json();
  if (!is_record(parsed)) {
    throw new Error('Unexpected MCP HTTP response: expected JSON object');
  }
  const rpc_response: JsonRpcResponse = {};
  if (typeof parsed.jsonrpc === 'string') rpc_response.jsonrpc = parsed.jsonrpc;
  if (typeof parsed.id === 'number') rpc_response.id = parsed.id;
  if ('result' in parsed) rpc_response.result = parsed.result;
  if ('error' in parsed) rpc_response.error = parsed.error;
  return rpc_response;
}

function timeout_signal(timeout_seconds: number): AbortSignal | undefined {
  if (!Number.isFinite(timeout_seconds) || timeout_seconds <= 0) return undefined;
  return AbortSignal.timeout(Math.ceil(timeout_seconds * 1000));
}

function with_optional_signal(init: Omit<RequestInit, 'signal'>, signal: AbortSignal | undefined): RequestInit {
  return signal === undefined ? init : { ...init, signal };
}

export abstract class BaseMCPClient implements MCPClient {
  request_id = 0;

  abstract start(): Promise<void>;
  abstract send_request(method: string, params?: Record<string, unknown> | null): Promise<Record<string, unknown>>;
  abstract close(): Promise<void>;
  protected abstract _send_initialized_notification(): Promise<void>;

  async initialize(): Promise<Record<string, unknown>> {
    const result = await this.send_request('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'gepa-mcp-adapter', version: '1.0' },
    });
    await this._send_initialized_notification();
    return result;
  }

  async list_tools(): Promise<MCPToolDefinition[]> {
    const result = await this.send_request('tools/list');
    return normalize_tools(result.tools);
  }

  async call_tool(name: string, arguments_: Record<string, unknown>): Promise<unknown> {
    return this.send_request('tools/call', { name, arguments: arguments_ });
  }
}

export class StdioMCPClient extends BaseMCPClient {
  readonly command: string;
  readonly args: string[];
  private process: ChildProcessWithoutNullStreams | null = null;
  private lines: ReturnType<typeof createInterface> | null = null;
  private pending = new Map<JsonRpcId, {
    resolve: (value: Record<string, unknown>) => void;
    reject: (reason: Error) => void;
  }>();

  constructor(command: string, args: string[] = []) {
    super();
    this.command = command;
    this.args = args;
  }

  async start(): Promise<void> {
    if (this.process !== null) return;

    const child = spawn(this.command, this.args, { stdio: ['pipe', 'pipe', 'pipe'] });
    this.process = child;
    const lines = createInterface({ input: child.stdout });
    this.lines = lines;

    lines.on('line', (line) => {
      this.handle_line(line);
    });

    child.on('error', (error) => {
      this.reject_all(error instanceof Error ? error : new Error(String(error)));
    });

    child.on('exit', (code, signal) => {
      if (this.pending.size > 0) {
        this.reject_all(new Error(`MCP stdio process exited before responding (code=${code}, signal=${signal})`));
      }
    });
  }

  async send_request(method: string, params: Record<string, unknown> | null = null): Promise<Record<string, unknown>> {
    if (this.process === null) {
      throw new RuntimeError('Process not started or streams not available');
    }

    this.request_id += 1;
    const id = this.request_id;
    const request: JsonRpcRequest = { jsonrpc: '2.0', method, id };
    if (params !== null) request.params = params;

    const response = new Promise<Record<string, unknown>>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });

    this.process.stdin.write(`${JSON.stringify(request)}\n`);
    return response;
  }

  protected async _send_initialized_notification(): Promise<void> {
    if (this.process === null) {
      throw new RuntimeError('Process not started or stdin not available');
    }
    const notification: JsonRpcRequest = { jsonrpc: '2.0', method: 'notifications/initialized' };
    this.process.stdin.write(`${JSON.stringify(notification)}\n`);
  }

  async close(): Promise<void> {
    if (this.lines !== null) {
      this.lines.close();
      this.lines = null;
    }
    if (this.process !== null) {
      this.process.stdin.end();
      this.process.kill();
      this.process = null;
    }
    this.reject_all(new Error('MCP stdio connection closed'));
  }

  private handle_line(line: string): void {
    let response: JsonRpcResponse;
    try {
      response = parse_json_rpc_response(line);
    } catch (error) {
      this.reject_all(error instanceof Error ? error : new Error(String(error)));
      return;
    }

    if (typeof response.id !== 'number') return;
    const pending = this.pending.get(response.id);
    if (pending === undefined) return;
    this.pending.delete(response.id);

    if (response.error !== undefined) {
      pending.reject(new Error(`MCP error: ${JSON.stringify(response.error)}`));
      return;
    }
    pending.resolve(normalize_result_dict(response.result));
  }

  private reject_all(error: Error): void {
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
  }
}

class RuntimeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RuntimeError';
  }
}

export class SSEMCPClient extends BaseMCPClient {
  readonly url: string;
  readonly headers: Record<string, string>;
  readonly timeout: number;
  private controller: AbortController | null = null;
  private endpoint_url: string | null = null;
  private endpoint_ready: Promise<string> | null = null;
  private resolve_endpoint: ((url: string) => void) | null = null;
  private reject_endpoint: ((error: Error) => void) | null = null;
  private pending = new Map<JsonRpcId, {
    resolve: (value: Record<string, unknown>) => void;
    reject: (reason: Error) => void;
  }>();

  constructor(url: string, headers: Record<string, string> | null = null, timeout = 30) {
    super();
    this.url = url;
    this.headers = headers ?? {};
    this.timeout = timeout;
  }

  async start(): Promise<void> {
    if (this.controller !== null) return;

    this.endpoint_ready = new Promise<string>((resolve, reject) => {
      this.resolve_endpoint = resolve;
      this.reject_endpoint = reject;
    });
    const controller = new AbortController();
    this.controller = controller;

    const response = await fetch(this.url, with_optional_signal({
      method: 'GET',
      headers: {
        Accept: 'text/event-stream',
        ...this.headers,
      },
    }, controller.signal));

    if (!response.ok) {
      throw new Error(`MCP SSE error: ${response.status} ${response.statusText}`);
    }
    if (response.body === null) {
      throw new Error('MCP SSE response did not include a readable body');
    }

    void this.read_sse_stream(response.body, controller.signal).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        this.reject_endpoint?.(error instanceof Error ? error : new Error(String(error)));
        this.reject_all(error instanceof Error ? error : new Error(String(error)));
      }
    });

    await this.await_endpoint();
  }

  async send_request(method: string, params: Record<string, unknown> | null = null): Promise<Record<string, unknown>> {
    const endpoint = await this.await_endpoint();
    this.request_id += 1;
    const id = this.request_id;
    const request: JsonRpcRequest = { jsonrpc: '2.0', method, id };
    if (params !== null) request.params = params;

    const response = new Promise<Record<string, unknown>>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });

    try {
      await this.post_to_endpoint(endpoint, request);
    } catch (error) {
      this.pending.delete(id);
      throw error;
    }
    return response;
  }

  protected async _send_initialized_notification(): Promise<void> {
    const endpoint = await this.await_endpoint();
    await this.post_to_endpoint(endpoint, { jsonrpc: '2.0', method: 'notifications/initialized' });
  }

  async close(): Promise<void> {
    this.controller?.abort();
    this.controller = null;
    this.endpoint_url = null;
    this.endpoint_ready = null;
    this.resolve_endpoint = null;
    this.reject_endpoint = null;
    this.reject_all(new Error('MCP SSE connection closed'));
    return undefined;
  }

  private async await_endpoint(): Promise<string> {
    if (this.endpoint_url !== null) return this.endpoint_url;
    if (this.endpoint_ready === null) {
      throw new RuntimeError('SSE connection not started');
    }
    return this.endpoint_ready;
  }

  private async post_to_endpoint(endpoint: string, request: JsonRpcRequest): Promise<void> {
    const response = await fetch(endpoint, with_optional_signal({
      method: 'POST',
      headers: {
        Accept: 'application/json, text/event-stream',
        Connection: 'close',
        'Content-Type': 'application/json',
        ...this.headers,
      },
      body: JSON.stringify(request),
    }, timeout_signal(this.timeout)));
    if (!response.ok) {
      throw new Error(`MCP SSE POST error: ${response.status} ${response.statusText}`);
    }
  }

  private async read_sse_stream(stream: ReadableStream<Uint8Array>, signal: AbortSignal): Promise<void> {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    try {
      while (!signal.aborted) {
        const result = await reader.read();
        if (result.done) break;
        buffer += decoder.decode(result.value, { stream: true });
        let boundary = buffer.indexOf('\n\n');
        while (boundary >= 0) {
          const event_block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          this.handle_sse_event(event_block);
          boundary = buffer.indexOf('\n\n');
        }
      }
      const rest = buffer.trim();
      if (rest.length > 0) {
        this.handle_sse_event(rest);
      }
    } finally {
      reader.releaseLock();
    }
  }

  private handle_sse_event(event_block: string): void {
    let event_name = 'message';
    const data_lines: string[] = [];
    for (const line of event_block.split(/\r?\n/)) {
      if (line.startsWith('event:')) {
        event_name = line.slice('event:'.length).trim();
      } else if (line.startsWith('data:')) {
        data_lines.push(line.slice('data:'.length).trimStart());
      }
    }
    const data = data_lines.join('\n');
    if (data.length === 0) return;
    if (event_name === 'endpoint') {
      this.endpoint_url = new URL(data, this.url).toString();
      this.resolve_endpoint?.(this.endpoint_url);
      return;
    }
    if (event_name === 'message' || event_name === 'jsonrpc') {
      this.dispatch_response(parse_json_rpc_response(data));
    }
  }

  private dispatch_response(response: JsonRpcResponse): void {
    if (typeof response.id !== 'number') return;
    const pending = this.pending.get(response.id);
    if (pending === undefined) return;
    this.pending.delete(response.id);

    if (response.error !== undefined) {
      pending.reject(new Error(`MCP error: ${JSON.stringify(response.error)}`));
      return;
    }
    pending.resolve(normalize_result_dict(response.result));
  }

  private reject_all(error: Error): void {
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
  }
}

export class StreamableHTTPMCPClient extends BaseMCPClient {
  readonly url: string;
  readonly headers: Record<string, string>;
  readonly timeout: number;
  readonly sse_read_timeout: number;
  private started = false;

  constructor(url: string, headers: Record<string, string> | null = null, timeout = 30, sse_read_timeout = 300) {
    super();
    this.url = url;
    this.headers = headers ?? {};
    this.timeout = timeout;
    this.sse_read_timeout = sse_read_timeout;
  }

  async start(): Promise<void> {
    this.started = true;
  }

  async send_request(method: string, params: Record<string, unknown> | null = null): Promise<Record<string, unknown>> {
    if (!this.started) {
      throw new RuntimeError('StreamableHTTP connection not started');
    }
    this.request_id += 1;
    const request: JsonRpcRequest = { jsonrpc: '2.0', method, id: this.request_id };
    if (params !== null) request.params = params;

    const response = await fetch(this.url, with_optional_signal({
      method: 'POST',
      headers: {
        Accept: 'application/json, text/event-stream',
        Connection: 'close',
        'Content-Type': 'application/json',
        ...this.headers,
      },
      body: JSON.stringify(request),
    }, timeout_signal(this.timeout)));
    const rpc_response = await read_json_response(response);
    if (rpc_response.error !== undefined) {
      throw new Error(`MCP error: ${JSON.stringify(rpc_response.error)}`);
    }
    return normalize_result_dict(rpc_response.result);
  }

  protected async _send_initialized_notification(): Promise<void> {
    if (!this.started) {
      throw new RuntimeError('StreamableHTTP connection not started');
    }
    const notification: JsonRpcRequest = { jsonrpc: '2.0', method: 'notifications/initialized' };
    const response = await fetch(this.url, with_optional_signal({
      method: 'POST',
      headers: {
        Accept: 'application/json, text/event-stream',
        Connection: 'close',
        'Content-Type': 'application/json',
        ...this.headers,
      },
      body: JSON.stringify(notification),
    }, timeout_signal(this.timeout)));
    if (!response.ok) {
      throw new Error(`MCP HTTP error: ${response.status} ${response.statusText}`);
    }
  }

  async close(): Promise<void> {
    this.started = false;
    return undefined;
  }
}

export type CreateMCPClientOptions = {
  server_params?: StdioServerParameters | null;
  remote_url?: string | null;
  remote_transport?: string;
  remote_headers?: Record<string, string> | null;
  remote_timeout?: number;
  sse_read_timeout?: number;
};

export function create_mcp_client({
  server_params = null,
  remote_url = null,
  remote_transport = 'sse',
  remote_headers = null,
  remote_timeout = 30,
  sse_read_timeout = 300,
}: CreateMCPClientOptions): BaseMCPClient {
  if (server_params !== null && remote_url !== null) {
    throw new Error('Provide either server_params (local) or remote_url (remote), not both');
  }
  if (server_params === null && remote_url === null) {
    throw new Error('Must provide either server_params (local) or remote_url (remote)');
  }
  if (server_params !== null) {
    return new StdioMCPClient(server_params.command, server_params.args ?? []);
  }
  const remote = remote_url;
  if (remote === null) {
    throw new Error('Must provide either server_params (local) or remote_url (remote)');
  }
  if (remote_transport === 'sse') {
    return new SSEMCPClient(remote, remote_headers, remote_timeout);
  }
  if (remote_transport === 'streamable_http') {
    return new StreamableHTTPMCPClient(remote, remote_headers, remote_timeout, sse_read_timeout);
  }
  throw new Error(`Unknown remote transport: ${remote_transport}. Must be 'sse' or 'streamable_http'`);
}
