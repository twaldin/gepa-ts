import { describe, expect, it, vi } from 'vitest';
import {
  ChromaVectorStore,
  LanceDBVectorStore,
  MilvusVectorStore,
  QdrantVectorStore,
  WeaviateVectorStore,
} from '../../src/adapters/generic_rag_adapter/vector_stores/index.js';

describe('generic RAG vector stores', () => {
  it('wraps Chroma query results and metadata filters', () => {
    const query = vi.fn(() => ({
      documents: [['doc']],
      metadatas: [[{ source: 's' }]],
      distances: [[0.25]],
    }));
    const collection = { query, count: () => 1, peek: () => ({ embeddings: [[0.1, 0.2]] }) };
    const store = new ChromaVectorStore({ get_collection: () => collection }, 'docs');

    expect(store.similarity_search('hello', 3, { category: 'x' })).toEqual([
      { content: 'doc', metadata: { source: 's' }, score: 0.75 },
    ]);
    expect(query).toHaveBeenCalledWith({
      query_texts: ['hello'],
      n_results: 3,
      where: { category: { $eq: 'x' } },
      include: ['documents', 'metadatas', 'distances'],
    });
    expect(store.get_collection_info()).toMatchObject({ name: 'docs', document_count: 1, dimension: 2, vector_store_type: 'chromadb' });
  });

  it('wraps LanceDB table search, insert, and delete operations', () => {
    const where = vi.fn(function where_fn() {
      return query_builder;
    });
    const to_pandas = vi.fn(() => [{ content: 'doc', metadata: { id: 1 }, _distance: 0.2 }]);
    const query_builder = { where, to_pandas };
    const table = {
      search: vi.fn(() => ({ limit: vi.fn(() => query_builder) })),
      add: vi.fn(),
      delete: vi.fn(),
      count_rows: () => 5,
      schema: 'schema',
    };
    const store = new LanceDBVectorStore({ open_table: () => table }, 'docs', () => [0.1, 0.2]);

    expect(store.similarity_search('hello', 2, { source: 's' })).toEqual([
      { content: 'doc', metadata: { id: 1 }, score: 0.2 },
    ]);
    expect(table.search).toHaveBeenCalledWith([0.1, 0.2]);
    expect(store.add_documents([{ content: 'new' }], [[0.3]], ['id1'])).toEqual(['id1']);
    expect(table.add).toHaveBeenCalledWith([{ id: 'id1', vector: [0.3], content: 'new' }], { mode: 'append' });
    expect(store.delete_documents(['id1'])).toBe(true);
    expect(store.get_collection_info()).toMatchObject({ name: 'docs', document_count: 5, vector_store_type: 'lancedb' });
  });

  it('wraps Milvus search results and collection metadata', () => {
    const client = {
      has_collection: () => true,
      load_collection: vi.fn(),
      search: vi.fn(() => [[{ id: '1', distance: 0.9, entity: { content: 'doc', metadata: { kind: 'k' } } }]]),
      insert: vi.fn(() => ({ ids: ['id1'] })),
      delete: vi.fn(() => ({ delete_count: 1 })),
      describe_collection: () => ({ fields: [{ name: 'vector', type: 'FloatVector', params: { dim: 3 } }] }),
      get_collection_stats: () => ({ row_count: 7 }),
    };
    const store = new MilvusVectorStore(client, 'docs', () => [0, 0, 1]);

    expect(store.vector_search([1, 2, 3], 1, { source: 's' })).toEqual([
      { content: 'doc', metadata: { kind: 'k' }, score: 0.9 },
    ]);
    expect(client.search).toHaveBeenCalledWith({
      collection_name: 'docs',
      data: [[1, 2, 3]],
      limit: 1,
      filter: 'source == "s"',
      output_fields: ['*'],
    });
    expect(store.add_documents([{ content: 'x' }], [[0.1]])).toEqual(['id1']);
    expect(store.delete_documents(['id1'])).toBe(true);
    expect(store.get_collection_info()).toMatchObject({ document_count: 7, dimension: 3, vector_store_type: 'milvus' });
  });

  it('wraps Qdrant query, upsert, delete, and info calls', () => {
    const client = {
      get_collection: () => ({
        points_count: 9,
        config: { params: { vectors: { size: 4, distance: { name: 'Cosine' } } } },
        status: { name: 'GREEN' },
      }),
      query_points: vi.fn(() => ({ points: [{ score: 0.8, payload: { content: 'doc', metadata: { a: 1 } } }] })),
      upsert: vi.fn(() => ({ status: { name: 'COMPLETED' } })),
      delete: vi.fn(() => ({ status: { name: 'COMPLETED' } })),
    };
    const store = new QdrantVectorStore(client, 'docs', () => [1, 2]);

    expect(store.similarity_search('hello', 1)).toEqual([{ content: 'doc', metadata: { a: 1 }, score: 0.8 }]);
    expect(client.query_points).toHaveBeenCalledWith({
      collection_name: 'docs',
      query: [1, 2],
      query_filter: null,
      limit: 1,
      with_payload: true,
      with_vectors: false,
      score_threshold: null,
    });
    expect(store.add_documents([{ content: 'x' }], [[0.1]], ['id1'])).toEqual(['id1']);
    expect(store.delete_documents(['id1'])).toBe(true);
    expect(store.get_collection_info()).toMatchObject({ document_count: 9, dimension: 4, distance_metric: 'cosine', status: 'green' });
  });

  it('wraps Weaviate vector and hybrid search responses', () => {
    const collection = {
      query: {
        near_vector: vi.fn(() => ({ objects: [{ properties: { content: 'doc', metadata: { s: 1 } }, metadata: { score: 0.7 } }] })),
        hybrid: vi.fn(() => ({ objects: [{ properties: { content: 'hybrid' }, metadata: { score: 0.6 } }] })),
      },
      aggregate: { over_all: () => ({ total_count: 2 }) },
      config: { get: () => ({ properties: [{ name: 'content' }] }) },
    };
    const store = new WeaviateVectorStore({ collections: { get: () => collection } }, 'docs', () => [0.1]);

    expect(store.vector_search([0.1], 1)).toEqual([{ content: 'doc', metadata: { s: 1 }, score: 0.7 }]);
    expect(store.hybrid_search('hello', 1, 0.25)).toEqual([{ content: 'hybrid', metadata: {}, score: 0.6 }]);
    expect(store.get_collection_info()).toMatchObject({ name: 'docs', document_count: 2, vector_store_type: 'weaviate' });
    expect(store.supports_hybrid_search()).toBe(true);
  });
});
