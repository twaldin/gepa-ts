import { describe, expect, it, vi } from 'vitest';
import { RAGPipeline, VectorStoreInterface, type RAGDocument, type RAGMetadataFilter } from '../../src/index.js';

class MockVectorStore extends VectorStoreInterface {
  readonly documents: RAGDocument[] = [
    {
      id: 'doc1',
      content: 'Machine learning is a subset of artificial intelligence.',
      metadata: { category: 'AI', score: 0.95 },
    },
    {
      id: 'doc2',
      content: 'Python is a popular programming language for data science.',
      metadata: { category: 'programming', score: 0.89 },
    },
  ];

  similarity_search(_query: string, k: number = 5, _filters: RAGMetadataFilter | null = null): RAGDocument[] {
    return this.documents.slice(0, k);
  }

  vector_search(_query_vector: number[], k: number = 5, _filters: RAGMetadataFilter | null = null): RAGDocument[] {
    return this.documents.slice(0, k);
  }

  hybrid_search(_query: string, k: number = 5, _alpha: number = 0.5): RAGDocument[] {
    return this.documents.slice(0, k);
  }

  get_collection_info(): Record<string, unknown> {
    return { name: 'mock_collection', document_count: this.documents.length, vector_store_type: 'mock' };
  }
}

function make_pipeline(llm_client: (messages: Array<{ role: string; content: string }>) => string = () => 'This is a test response from the LLM.') {
  return new RAGPipeline({
    vector_store: new MockVectorStore(),
    llm_client,
    embedding_function: () => Array.from({ length: 384 }, () => 0.1),
  });
}

describe('RAGPipeline', () => {
  it('initializes with upstream-compatible defaults', () => {
    const vector_store = new MockVectorStore();
    const llm_client = () => 'ok';
    const pipeline = new RAGPipeline({ vector_store, llm_client });

    expect(pipeline.vector_store).toBe(vector_store);
    expect(pipeline.llm_client).toBe(llm_client);
    expect(pipeline.embedding_model).toBe('text-embedding-3-small');
    expect(pipeline.embedding_function).toBeTypeOf('function');
  });

  it('executes a basic RAG flow with metadata', async () => {
    const pipeline = make_pipeline();

    const result = await pipeline.execute_rag(
      'What is machine learning?',
      { answer_generation: 'Answer using context.' },
      { retrieval_strategy: 'similarity', top_k: 2 },
    );

    expect(result.original_query).toBe('What is machine learning?');
    expect(result.retrieved_docs).toHaveLength(2);
    expect(result.synthesized_context).toContain('[Document 1]');
    expect(result.generated_answer).toBe('This is a test response from the LLM.');
    expect(result.metadata).toMatchObject({ retrieval_count: 2, vector_store_type: 'mock' });
    expect(result.metadata.total_tokens).toBeGreaterThan(0);
  });

  it('supports query reformulation', async () => {
    const llm_client = vi.fn().mockReturnValueOnce('machine learning expanded').mockReturnValueOnce('answer');
    const pipeline = make_pipeline(llm_client);

    const result = await pipeline.execute_rag(
      'What is ML?',
      { query_reformulation: 'Reformulate this query', answer_generation: 'Answer.' },
      { retrieval_strategy: 'similarity', top_k: 1 },
    );

    expect(result.original_query).toBe('What is ML?');
    expect(result.reformulated_query).toBe('machine learning expanded');
    expect(llm_client).toHaveBeenCalledTimes(2);
  });

  it('supports similarity, vector, and hybrid retrieval strategies', async () => {
    const pipeline = make_pipeline();

    for (const strategy of ['similarity', 'vector', 'hybrid']) {
      const result = await pipeline.execute_rag(
        'test query',
        { answer_generation: 'Answer.' },
        { retrieval_strategy: strategy, top_k: 2 },
      );

      expect(result.retrieved_docs).toHaveLength(2);
      expect(result.generated_answer).toBeTruthy();
    }
  });

  it('reranks documents when the LLM returns a complete ranking', async () => {
    const llm_client = vi.fn().mockReturnValueOnce('2,1').mockReturnValueOnce('answer');
    const pipeline = make_pipeline(llm_client);

    const result = await pipeline.execute_rag(
      'machine learning',
      { reranking_criteria: 'Rank documents', answer_generation: 'Answer.' },
      { retrieval_strategy: 'similarity', top_k: 2 },
    );

    expect(result.retrieved_docs.map((doc) => doc.id)).toEqual(['doc2', 'doc1']);
  });

  it('falls back cleanly when LLM calls fail', async () => {
    const pipeline = make_pipeline(() => {
      throw new Error('LLM Error');
    });

    const result = await pipeline.execute_rag(
      'test query',
      { query_reformulation: 'Reformulate.', answer_generation: 'Answer.' },
      { retrieval_strategy: 'similarity', top_k: 1 },
    );

    expect(result.reformulated_query).toBe('test query');
    expect(result.generated_answer).toBe('Error generating answer: LLM Error');
  });

  it('throws for unknown retrieval strategies', () => {
    const pipeline = make_pipeline();

    expect(() => pipeline._retrieve_documents('query', { retrieval_strategy: 'unknown' })).toThrow(
      'Unknown retrieval strategy: unknown',
    );
  });
});
