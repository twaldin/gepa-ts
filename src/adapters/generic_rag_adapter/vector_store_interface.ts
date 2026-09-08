export type RAGDocument = {
  content: string;
  metadata?: Record<string, unknown>;
  score?: number;
  [key: string]: unknown;
};

export type RAGMetadataFilter = Record<string, unknown>;

export abstract class VectorStoreInterface {
  abstract similarity_search(
    query: string,
    k?: number,
    filters?: RAGMetadataFilter | null,
  ): RAGDocument[];

  abstract vector_search(
    query_vector: number[],
    k?: number,
    filters?: RAGMetadataFilter | null,
  ): RAGDocument[];

  hybrid_search(
    query: string,
    k: number = 5,
    _alpha: number = 0.5,
    filters: RAGMetadataFilter | null = null,
  ): RAGDocument[] {
    return this.similarity_search(query, k, filters);
  }

  abstract get_collection_info(): Record<string, unknown>;

  get_embedding_dimension(): number {
    const dimension = this.get_collection_info()['dimension'];
    return typeof dimension === 'number' ? dimension : 0;
  }

  supports_hybrid_search(): boolean {
    return false;
  }

  supports_metadata_filtering(): boolean {
    return true;
  }
}
