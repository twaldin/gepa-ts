import { describe, expect, it } from 'vitest';
import { createServer, type ServerResponse } from 'node:http';
import { MCPAdapter, SSEMCPClient, StdioMCPClient, StreamableHTTPMCPClient, create_mcp_client } from '../../src/adapters/mcp_adapter/index.js';

describe('MCPAdapter', () => {
  it('runs a two-pass tool workflow with an injected client factory', async () => {
    const calls: Array<{ tool: string; args: Record<string, unknown> }> = [];
    const model_messages: unknown[] = [];
    const adapter = new MCPAdapter({
      tool_names: 'lookup',
      task_model: async (messages) => {
        model_messages.push(messages);
        return model_messages.length === 1
          ? JSON.stringify({ action: 'call_tool', tool: 'lookup', arguments: { id: '42' } })
          : 'final answer from tool';
      },
      metric_fn: (_item, output) => (output.includes('final') ? 1 : 0),
      client_factory: () => ({
        start: async () => undefined,
        initialize: async () => ({ serverInfo: { name: 'test-server' } }),
        list_tools: async () => [
          {
            name: 'lookup',
            description: 'Look up facts',
            inputSchema: { properties: { id: { type: 'string' } } },
          },
        ],
        call_tool: async (tool, args) => {
          calls.push({ tool, args });
          return { content: [{ type: 'text', text: 'tool says 42' }] };
        },
        close: async () => undefined,
      }),
    });

    const result = await adapter.evaluate(
      [{ user_query: 'What is 42?', tool_arguments: { id: '42' }, reference_answer: '42', additional_context: {} }],
      { tool_description: 'Use this for exact lookup.', system_prompt: 'You use tools.' },
      true,
    );

    expect(calls).toEqual([{ tool: 'lookup', args: { id: '42' } }]);
    expect(model_messages).toHaveLength(2);
    expect(result.outputs).toEqual([
      {
        final_answer: 'final answer from tool',
        tool_called: true,
        selected_tool: 'lookup',
        tool_response: 'tool says 42',
      },
    ]);
    expect(result.scores).toEqual([1]);
    expect(result.trajectories?.[0]?.tool_description_used).toBe('Use this for exact lookup.');
  });

  it('returns failure scores when configured tools are unavailable', async () => {
    const adapter = new MCPAdapter({
      tool_names: ['missing'],
      task_model: async () => 'unused',
      metric_fn: () => 1,
      failure_score: -1,
      client_factory: () => ({
        start: async () => undefined,
        initialize: async () => ({}),
        list_tools: async () => [{ name: 'other', description: 'Other tool', inputSchema: {} }],
        call_tool: async () => ({ content: [] }),
        close: async () => undefined,
      }),
    });

    const result = await adapter.evaluate(
      [{ user_query: 'query', tool_arguments: {}, reference_answer: null, additional_context: {} }],
      { tool_description: 'desc' },
      true,
    );

    expect(result.scores).toEqual([-1]);
    expect(result.outputs[0]?.tool_called).toBe(false);
    expect(result.trajectories?.[0]?.model_first_pass_output).toContain('SESSION ERROR:');
  });

  it('builds reflective data for tool descriptions and system prompts', async () => {
    const adapter = new MCPAdapter({
      tool_names: ['search', 'lookup'],
      task_model: async () => 'unused',
      metric_fn: () => 0,
      client_factory: () => {
        throw new Error('not used');
      },
    });

    const eval_batch = {
      outputs: [
        { final_answer: 'bad', tool_called: true, selected_tool: 'search', tool_response: 'nope' },
      ],
      scores: [0.25],
      trajectories: [
        {
          user_query: 'Find it',
          tool_names: ['search', 'lookup'],
          selected_tool: 'search',
          tool_called: true,
          tool_arguments: { q: 'Find it' },
          tool_response: 'nope',
          tool_description_used: 'Searches content',
          system_prompt_used: 'Use tools',
          model_first_pass_output: '{"action":"call_tool"}',
          model_final_output: 'bad',
          score: 0.25,
        },
      ],
    };

    const reflective = adapter.make_reflective_dataset(
      { tool_description: 'Searches content', system_prompt: 'Use tools' },
      eval_batch,
      ['tool_description', 'system_prompt'],
    );

    expect(reflective.tool_description?.[0]?.Feedback).toContain("Tool 'search' was called");
    expect(reflective.system_prompt?.[0]).toEqual({
      Inputs: { user_query: 'Find it', system_prompt: 'Use tools' },
      'Generated Outputs': 'bad',
      Feedback: 'System prompt may need improvement (score: 0.25). Model called tool, but answer was incorrect.',
    });
  });

  it('creates a stdio MCP client and exchanges upstream-shaped JSON-RPC messages', async () => {
    const server = `
const readline = require('node:readline');
const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  const req = JSON.parse(line);
  if (!req.id) return;
  if (req.method === 'initialize') {
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: req.id, result: { serverInfo: { name: 'fake' } } }) + '\\n');
  } else if (req.method === 'tools/list') {
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: req.id, result: { tools: [{ name: 'lookup', description: 'Look up values', inputSchema: {} }] } }) + '\\n');
  } else if (req.method === 'tools/call') {
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: req.id, result: { content: [{ type: 'text', text: req.params.arguments.id }] } }) + '\\n');
  }
});
`;
    const client = new StdioMCPClient(process.execPath, ['-e', server]);
    await client.start();
    await expect(client.initialize()).resolves.toEqual({ serverInfo: { name: 'fake' } });
    await expect(client.list_tools()).resolves.toEqual([
      { name: 'lookup', description: 'Look up values', inputSchema: {} },
    ]);
    await expect(client.call_tool('lookup', { id: '42' })).resolves.toEqual({
      content: [{ type: 'text', text: '42' }],
    });
    await client.close();
  });

  it('wires MCPAdapter to the native stdio client when server_params are provided', async () => {
    const server = `
const readline = require('node:readline');
const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  const req = JSON.parse(line);
  if (!req.id) return;
  if (req.method === 'initialize') {
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: req.id, result: {} }) + '\\n');
  } else if (req.method === 'tools/list') {
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: req.id, result: { tools: [{ name: 'lookup', description: 'Look up values', inputSchema: { properties: { id: { type: 'string' } } } }] } }) + '\\n');
  } else if (req.method === 'tools/call') {
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: req.id, result: { content: [{ type: 'text', text: 'tool ' + req.params.arguments.id }] } }) + '\\n');
  }
});
`;
    let model_call_count = 0;
    const adapter = new MCPAdapter({
      tool_names: 'lookup',
      task_model: async () => {
        model_call_count += 1;
        return model_call_count === 1
        ? JSON.stringify({ action: 'call_tool', tool: 'lookup', arguments: { id: '7' } })
        : 'final from stdio';
      },
      metric_fn: (_item, output) => (output === 'final from stdio' ? 1 : 0),
      server_params: { command: process.execPath, args: ['-e', server] },
    });

    const result = await adapter.evaluate(
      [{ user_query: 'Use lookup', tool_arguments: { id: '7' }, reference_answer: '7', additional_context: {} }],
      { tool_description: 'Use lookup.', system_prompt: 'Use tools.' },
      true,
    );

    expect(result.scores).toEqual([1]);
    expect(result.outputs[0]).toEqual({
      final_answer: 'final from stdio',
      tool_called: true,
      selected_tool: 'lookup',
      tool_response: 'tool 7',
    });
  });

  it('validates MCP client factory configuration like upstream', () => {
    expect(() => create_mcp_client({})).toThrow(/Must provide either server_params/);
    expect(() => create_mcp_client({
      server_params: { command: process.execPath },
      remote_url: 'https://example.test/mcp',
    })).toThrow(/Provide either server_params/);
    expect(() => create_mcp_client({
      remote_url: 'https://example.test/mcp',
      remote_transport: 'websocket',
    })).toThrow(/Unknown remote transport/);
  });

  it('exchanges JSON-RPC requests through the native streamable HTTP MCP client', async () => {
    const seen_methods: string[] = [];
    const server = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', (chunk: Buffer) => chunks.push(chunk));
      req.on('end', () => {
        const body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as {
          id?: number;
          method?: string;
          params?: { arguments?: { id?: string } };
        };
        seen_methods.push(body.method ?? '');
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Connection', 'close');
        if (body.method === 'notifications/initialized') {
          res.statusCode = 202;
          res.end(JSON.stringify({}));
        } else if (body.method === 'initialize') {
          res.end(JSON.stringify({ jsonrpc: '2.0', id: body.id, result: { serverInfo: { name: 'http' } } }));
        } else if (body.method === 'tools/list') {
          res.end(JSON.stringify({ jsonrpc: '2.0', id: body.id, result: { tools: [{ name: 'lookup', description: 'HTTP lookup', inputSchema: {} }] } }));
        } else if (body.method === 'tools/call') {
          res.end(JSON.stringify({ jsonrpc: '2.0', id: body.id, result: { content: [{ type: 'text', text: body.params?.arguments?.id ?? '' }] } }));
        } else {
          res.end(JSON.stringify({ jsonrpc: '2.0', id: body.id, error: { message: 'unknown' } }));
        }
      });
    });

    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const address = server.address();
    if (address === null || typeof address === 'string') {
      server.close();
      throw new Error('Expected HTTP server address');
    }

    const client = new StreamableHTTPMCPClient(`http://127.0.0.1:${address.port}/mcp`);
    try {
      await client.start();
      await expect(client.initialize()).resolves.toEqual({ serverInfo: { name: 'http' } });
      await expect(client.list_tools()).resolves.toEqual([
        { name: 'lookup', description: 'HTTP lookup', inputSchema: {} },
      ]);
      await expect(client.call_tool('lookup', { id: '99' })).resolves.toEqual({
        content: [{ type: 'text', text: '99' }],
      });
      expect(seen_methods).toEqual(['initialize', 'notifications/initialized', 'tools/list', 'tools/call']);
    } finally {
      await client.close();
      await new Promise<void>((resolve, reject) => {
        server.close((error) => {
          if (error && 'code' in error && error.code === 'ERR_SERVER_NOT_RUNNING') resolve();
          else if (error) reject(error);
          else resolve();
        });
      });
      server.closeAllConnections();
    }
  });

  it('exchanges JSON-RPC requests through the native SSE MCP client', async () => {
    const seen_methods: string[] = [];
    let sse_response: ServerResponse | null = null;
    const server = createServer((req, res) => {
      if (req.method === 'GET' && req.url === '/sse') {
        sse_response = res;
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          Connection: 'keep-alive',
          'Cache-Control': 'no-cache',
        });
        res.write('event: endpoint\ndata: /messages\n\n');
        return;
      }

      const chunks: Buffer[] = [];
      req.on('data', (chunk: Buffer) => chunks.push(chunk));
      req.on('end', () => {
        const body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as {
          id?: number;
          method?: string;
          params?: { arguments?: { id?: string } };
        };
        seen_methods.push(body.method ?? '');
        res.setHeader('Connection', 'close');
        res.statusCode = 202;
        res.end();

        if (body.id !== undefined && sse_response !== null) {
          if (body.method === 'initialize') {
            sse_response.write(`event: message\ndata: ${JSON.stringify({ jsonrpc: '2.0', id: body.id, result: { serverInfo: { name: 'sse' } } })}\n\n`);
          } else if (body.method === 'tools/list') {
            sse_response.write(`event: message\ndata: ${JSON.stringify({ jsonrpc: '2.0', id: body.id, result: { tools: [{ name: 'lookup', description: 'SSE lookup', inputSchema: {} }] } })}\n\n`);
          } else if (body.method === 'tools/call') {
            sse_response.write(`event: message\ndata: ${JSON.stringify({ jsonrpc: '2.0', id: body.id, result: { content: [{ type: 'text', text: body.params?.arguments?.id ?? '' }] } })}\n\n`);
          }
        }
      });
    });

    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const address = server.address();
    if (address === null || typeof address === 'string') {
      server.close();
      throw new Error('Expected HTTP server address');
    }

    const client = new SSEMCPClient(`http://127.0.0.1:${address.port}/sse`);
    try {
      await client.start();
      await expect(client.initialize()).resolves.toEqual({ serverInfo: { name: 'sse' } });
      await expect(client.list_tools()).resolves.toEqual([
        { name: 'lookup', description: 'SSE lookup', inputSchema: {} },
      ]);
      await expect(client.call_tool('lookup', { id: '77' })).resolves.toEqual({
        content: [{ type: 'text', text: '77' }],
      });
      expect(seen_methods).toEqual(['initialize', 'notifications/initialized', 'tools/list', 'tools/call']);
    } finally {
      sse_response?.end();
      await client.close();
      await new Promise<void>((resolve, reject) => {
        server.close((error) => {
          if (error && 'code' in error && error.code === 'ERR_SERVER_NOT_RUNNING') resolve();
          else if (error) reject(error);
          else resolve();
        });
      });
      server.closeAllConnections();
    }
  });
});
