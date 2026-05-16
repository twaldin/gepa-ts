import { describe, expect, it } from 'vitest';
import {
  GenericRAGAdapter,
  VectorStoreInterface,
  type RAGDataInst,
  type RAGDocument,
  type RAGMetadataFilter,
} from '../../src/index.js';

class MockVectorStore extends VectorStoreInterface {
  readonly documents: RAGDocument[] = [
    {
      id: 'doc1',
      content: 'Machine learning is a subset of artificial intelligence.',
      metadata: { doc_id: 'doc1', category: 'AI' },
    },
    {
      id: 'doc2',
      content: 'Python is a popular programming language for data science.',
      metadata: { doc_id: 'doc2', category: 'programming' },
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
    return { name: 'test_collection', document_count: this.documents.length, vector_store_type: 'mock' };
  }
}

function sample_data(): RAGDataInst[] {
  return [
    {
      query: 'What is machine learning?',
      ground_truth_answer: 'Machine learning is a subset of AI.',
      relevant_doc_ids: ['doc1'],
      metadata: { difficulty: 'beginner' },
    },
    {
      query: 'What programming language is used for ML?',
      ground_truth_answer: 'Python is commonly used for ML.',
      relevant_doc_ids: ['doc2'],
      metadata: { difficulty: 'beginner' },
    },
  ];
}

function make_adapter(response: string = 'Machine learning is a subset of AI.') {
  return new GenericRAGAdapter({
    vector_store: new MockVectorStore(),
    llm_model: () => response,
    embedding_function: () => Array.from({ length: 8 }, () => 0.1),
    rag_config: { retrieval_strategy: 'similarity', top_k: 2, retrieval_weight: 0.3, generation_weight: 0.7 },
  });
}

describe('GenericRAGAdapter', () => {
  it('initializes with upstream-compatible defaults', () => {
    const vector_store = new MockVectorStore();
    const adapter = new GenericRAGAdapter({
      vector_store,
      llm_model: () => 'answer',
      embedding_function: () => [0.1],
    });

    expect(adapter.vector_store).toBe(vector_store);
    expect(adapter.config).toMatchObject({
      retrieval_strategy: 'similarity',
      top_k: 5,
      retrieval_weight: 0.3,
      generation_weight: 0.7,
      hybrid_alpha: 0.5,
      filters: null,
    });
    expect(adapter.rag_pipeline).toBeTruthy();
    expect(adapter.evaluator).toBeTruthy();
  });

  it('evaluates a batch and returns scores and outputs', async () => {
    const adapter = make_adapter();

    const result = await adapter.evaluate(sample_data(), { answer_generation: 'Answer: {query}' });

    expect(result.outputs).toHaveLength(2);
    expect(result.scores).toHaveLength(2);
    expect(result.num_metric_calls).toBe(2);
    expect(result.trajectories).toBeUndefined();
    expect(result.outputs[0]?.final_answer).toBe('Machine learning is a subset of AI.');
    expect(result.scores.every((score) => score >= 0 && score <= 1)).toBe(true);
  });

  it('captures trajectories with retrieval and generation metrics', async () => {
    const adapter = make_adapter('Test answer');

    const result = await adapter.evaluate(sample_data(), { answer_generation: 'Answer: {query}' }, true);

    expect(result.trajectories).toHaveLength(2);
    const trajectory = result.trajectories?.[0];
    expect(trajectory?.original_query).toBe('What is machine learning?');
    expect(trajectory?.retrieved_docs).toHaveLength(2);
    expect(trajectory?.execution_metadata).toHaveProperty('retrieval_metrics');
    expect(trajectory?.execution_metadata).toHaveProperty('generation_metrics');
    expect(trajectory?.execution_metadata).toHaveProperty('overall_score');
  });

  it('supports vector and hybrid retrieval strategies through config', async () => {
    for (const strategy of ['vector', 'hybrid']) {
      const adapter = new GenericRAGAdapter({
        vector_store: new MockVectorStore(),
        llm_model: () => 'Test response',
        embedding_function: () => [0.1, 0.1],
        rag_config: { retrieval_strategy: strategy, top_k: 1 },
      });

      const result = await adapter.evaluate([sample_data()[0]!], { answer_generation: 'Answer.' });
      expect(result.outputs).toHaveLength(1);
      expect(result.scores[0]).toBeTypeOf('number');
    }
  });

  it('returns failure score and error trajectory for per-example failures', async () => {
    const adapter = new GenericRAGAdapter({
      vector_store: new MockVectorStore(),
      llm_model: () => 'answer',
      embedding_function: () => [0.1],
      rag_config: { retrieval_strategy: 'unknown', top_k: 1 },
      failure_score: 0.12,
    });

    const result = await adapter.evaluate([sample_data()[0]!], { answer_generation: 'Answer.' }, true);

    expect(result.scores).toEqual([0.12]);
    expect(result.outputs[0]?.final_answer).toBe('Error: Unknown retrieval strategy: unknown');
    expect(result.trajectories?.[0]?.execution_metadata).toEqual({ error: 'Unknown retrieval strategy: unknown' });
  });

  it('builds reflective datasets for answer generation', async () => {
    const adapter = make_adapter('Machine learning is a subset of AI.');
    const candidate = { answer_generation: 'Answer: {query}' };
    const eval_batch = await adapter.evaluate(sample_data(), candidate, true);

    const reflective_data = adapter.make_reflective_dataset(candidate, eval_batch, ['answer_generation']);

    expect(reflective_data['answer_generation']).toHaveLength(2);
    const example = reflective_data['answer_generation']?.[0];
    expect(example?.['Inputs']).toMatchObject({
      query: 'What is machine learning?',
      current_prompt: 'Answer: {query}',
    });
    expect(example?.['Generated Outputs']).toBe('Machine learning is a subset of AI.');
    expect(String(example?.['Feedback'])).toContain('answer');
  });

  it('builds component-specific reflective examples', async () => {
    const adapter = make_adapter('answer');
    const candidate = {
      query_reformulation: 'Reformulate.',
      context_synthesis: 'Synthesize.',
      reranking_criteria: 'Rank.',
      answer_generation: 'Answer.',
    };
    const eval_batch = await adapter.evaluate([sample_data()[0]!], candidate, true);

    const reflective_data = adapter.make_reflective_dataset(
      candidate,
      eval_batch,
      ['query_reformulation', 'context_synthesis', 'reranking_criteria'],
    );

    expect(reflective_data['query_reformulation']?.[0]?.['Inputs']).toMatchObject({
      original_query: 'What is machine learning?',
      current_prompt: 'Reformulate.',
    });
    expect(reflective_data['context_synthesis']?.[0]?.['Inputs']).toHaveProperty('retrieved_docs');
    expect(reflective_data['reranking_criteria']?.[0]?.['Generated Outputs']).toBe('Document ranking applied');
  });
});
