import { type RAGDocument, type RAGMetadataFilter, VectorStoreInterface } from './vector_store_interface.js';

export type RAGChatMessage = {
  role: 'system' | 'user' | 'assistant';
  content: string;
};

export type RAGCallableClient = (messages: RAGChatMessage[]) => string | Promise<string>;

export type RAGCompletionClient = {
  completion(args: { messages: RAGChatMessage[] }): {
    choices?: Array<{ message?: { content?: unknown } }>;
  } | Promise<{
    choices?: Array<{ message?: { content?: unknown } }>;
  }>;
};

export type RAGLLMClient = RAGCallableClient | RAGCompletionClient;

export type RAGPipelineConfig = {
  retrieval_strategy?: 'similarity' | 'hybrid' | 'vector' | string;
  top_k?: number;
  filters?: RAGMetadataFilter | null;
  hybrid_alpha?: number;
  [key: string]: unknown;
};

export type RAGPipelineResult = {
  original_query: string;
  reformulated_query: string;
  retrieved_docs: RAGDocument[];
  synthesized_context: string;
  generated_answer: string;
  metadata: {
    retrieval_count: number;
    total_tokens: number;
    vector_store_type: unknown;
  };
};

function is_completion_client(client: RAGLLMClient): client is RAGCompletionClient {
  return typeof client === 'object' && client !== null && typeof client.completion === 'function';
}

function response_to_string(response: string | { choices?: Array<{ message?: { content?: unknown } }> }): string {
  if (typeof response === 'string') return response;
  const content = response.choices?.[0]?.message?.content;
  return typeof content === 'string' ? content : '';
}

export class RAGPipeline {
  readonly vector_store: VectorStoreInterface;
  llm_client: RAGLLMClient;
  readonly embedding_model: string;
  embedding_function: (text: string) => number[];

  constructor({
    vector_store,
    llm_client,
    embedding_model = 'text-embedding-3-small',
    embedding_function = null,
  }: {
    vector_store: VectorStoreInterface;
    llm_client: RAGLLMClient;
    embedding_model?: string;
    embedding_function?: ((text: string) => number[]) | null;
  }) {
    this.vector_store = vector_store;
    this.llm_client = llm_client;
    this.embedding_model = embedding_model;
    this.embedding_function = embedding_function ?? this._default_embedding_function.bind(this);
  }

  async execute_rag(
    query: string,
    prompts: Record<string, string>,
    config: RAGPipelineConfig,
  ): Promise<RAGPipelineResult> {
    let reformulated_query = query;
    if (prompts['query_reformulation']?.trim()) {
      reformulated_query = await this._reformulate_query(query, prompts['query_reformulation']);
    }

    let retrieved_docs = this._retrieve_documents(reformulated_query, config);

    if (prompts['reranking_criteria']?.trim()) {
      retrieved_docs = await this._rerank_documents(retrieved_docs, query, prompts['reranking_criteria'], config);
    }

    const context = await this._synthesize_context(retrieved_docs, query, prompts['context_synthesis'] ?? '');
    const answer = await this._generate_answer(query, context, prompts['answer_generation'] ?? '');

    return {
      original_query: query,
      reformulated_query,
      retrieved_docs,
      synthesized_context: context,
      generated_answer: answer,
      metadata: {
        retrieval_count: retrieved_docs.length,
        total_tokens: this._estimate_token_count(context + answer),
        vector_store_type: this.vector_store.get_collection_info()['vector_store_type'] ?? 'unknown',
      },
    };
  }

  async _reformulate_query(query: string, reformulation_prompt: string): Promise<string> {
    const messages: RAGChatMessage[] = [
      { role: 'system', content: reformulation_prompt },
      { role: 'user', content: `Original query: ${query}` },
    ];

    try {
      const response = await this._call_llm(messages);
      const trimmed = response.trim();
      return trimmed || query;
    } catch {
      return query;
    }
  }

  _retrieve_documents(query: string, config: RAGPipelineConfig): RAGDocument[] {
    const retrieval_strategy = config.retrieval_strategy ?? 'similarity';
    const k = config.top_k ?? 5;
    const filters = config.filters ?? null;

    if (retrieval_strategy === 'similarity') {
      return this.vector_store.similarity_search(query, k, filters);
    }
    if (retrieval_strategy === 'hybrid') {
      if (this.vector_store.supports_hybrid_search()) {
        return this.vector_store.hybrid_search(query, k, config.hybrid_alpha ?? 0.5, filters);
      }
      return this.vector_store.similarity_search(query, k, filters);
    }
    if (retrieval_strategy === 'vector') {
      return this.vector_store.vector_search(this.embedding_function(query), k, filters);
    }

    throw new Error(`Unknown retrieval strategy: ${retrieval_strategy}`);
  }

  async _rerank_documents(
    documents: RAGDocument[],
    query: string,
    reranking_prompt: string,
    _config: RAGPipelineConfig,
  ): Promise<RAGDocument[]> {
    if (documents.length === 0) return documents;

    try {
      const doc_context = documents
        .map((doc, i) => `Document ${i + 1}: ${doc.content}`)
        .join('\n\n');
      const messages: RAGChatMessage[] = [
        { role: 'system', content: reranking_prompt },
        {
          role: 'user',
          content:
            `Query: ${query}\n\nDocuments:\n${doc_context}\n\n` +
            "Please rank these documents by relevance (return document numbers in order, e.g., '3,1,4,2,5'):",
        },
      ];

      const ranking_str = (await this._call_llm(messages)).trim();
      const rankings = ranking_str
        .split(',')
        .map((item) => Number.parseInt(item.trim(), 10) - 1)
        .filter((idx) => Number.isInteger(idx));

      if (rankings.length === documents.length) {
        const reranked = rankings
          .filter((idx) => idx >= 0 && idx < documents.length)
          .map((idx) => documents[idx])
          .filter((doc): doc is RAGDocument => doc !== undefined);
        if (reranked.length === documents.length) return reranked;
      }
    } catch {
      return documents;
    }

    return documents;
  }

  async _synthesize_context(documents: RAGDocument[], query: string, synthesis_prompt: string): Promise<string> {
    if (documents.length === 0) return '';

    if (!synthesis_prompt.trim()) {
      return documents
        .map((doc, i) => `[Document ${i + 1}] ${doc.content}`)
        .join('\n\n');
    }

    const doc_context = documents
      .map((doc, i) => `Document ${i + 1}: ${doc.content}`)
      .join('\n\n');
    const messages: RAGChatMessage[] = [
      { role: 'system', content: synthesis_prompt },
      { role: 'user', content: `Query: ${query}\n\nRetrieved Documents:\n${doc_context}` },
    ];

    try {
      const response = await this._call_llm(messages);
      return response.trim() || doc_context;
    } catch {
      return doc_context;
    }
  }

  async _generate_answer(query: string, context: string, generation_prompt: string): Promise<string> {
    const system_prompt = generation_prompt.trim()
      ? generation_prompt
      : "You are a helpful assistant. Answer the user's question based on the provided context.";
    const messages: RAGChatMessage[] = [
      { role: 'system', content: system_prompt },
      { role: 'user', content: `Context:\n${context}\n\nQuestion: ${query}` },
    ];

    try {
      const response = await this._call_llm(messages);
      return response.trim() || "I couldn't generate an answer based on the provided context.";
    } catch (e) {
      return `Error generating answer: ${String(e instanceof Error ? e.message : e)}`;
    }
  }

  _default_embedding_function(_text: string): number[] {
    throw new Error(
      `Failed to generate embeddings: no embedding_function was provided for ${this.embedding_model}`,
    );
  }

  _estimate_token_count(text: string): number {
    return Math.floor(text.length / 4);
  }

  private async _call_llm(messages: RAGChatMessage[]): Promise<string> {
    if (is_completion_client(this.llm_client)) {
      return response_to_string(await this.llm_client.completion({ messages }));
    }
    return response_to_string(await this.llm_client(messages));
  }
}
