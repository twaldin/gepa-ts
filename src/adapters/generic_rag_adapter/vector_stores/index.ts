import { VectorStoreInterface, type RAGDocument, type RAGMetadataFilter } from '../vector_store_interface.js';

export type EmbeddingFunction = (query: string) => number[] | { tolist?: () => number[] };

function is_record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function to_vector(value: number[] | { tolist?: () => number[] }): number[] {
  if (Array.isArray(value)) return value;
  if (typeof value.tolist === 'function') return value.tolist();
  return [];
}

function require_embedding(embedding_function: EmbeddingFunction | null, query: string): number[] {
  if (embedding_function === null) {
    throw new Error('No embedding function provided for similarity search');
  }
  try {
    return to_vector(embedding_function(query));
  } catch (error) {
    throw new Error(`Failed to compute embeddings for query: ${String(error)}`);
  }
}

function numeric(value: unknown, fallback: number = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function string_value(value: unknown, fallback: string = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function metadata(value: unknown): Record<string, unknown> {
  return is_record(value) ? value : {};
}

function equality_filter(filters: RAGMetadataFilter | null | undefined): Record<string, unknown> | null {
  if (!filters) return null;
  const converted: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(filters)) {
    converted[key] = is_record(value) ? value : { $eq: value };
  }
  return converted;
}

export class ChromaVectorStore extends VectorStoreInterface {
  readonly client: {
    get_collection?: (args: { name: string; embedding_function?: EmbeddingFunction | null }) => unknown;
    create_collection?: (args: { name: string; embedding_function?: EmbeddingFunction | null }) => unknown;
  };
  readonly collection_name: string;
  readonly embedding_function: EmbeddingFunction | null;
  readonly collection: {
    query: (args: Record<string, unknown>) => Record<string, unknown>;
    count?: () => number;
    peek?: (args: { limit: number }) => Record<string, unknown>;
  };

  constructor(client: ChromaVectorStore['client'], collection_name: string, embedding_function: EmbeddingFunction | null = null) {
    super();
    this.client = client;
    this.collection_name = collection_name;
    this.embedding_function = embedding_function;
    try {
      this.collection = client.get_collection?.({ name: collection_name, embedding_function }) as ChromaVectorStore['collection'];
    } catch {
      this.collection = undefined as never;
    }
    if (this.collection === undefined) {
      const created = client.create_collection?.({ name: collection_name, embedding_function });
      if (!is_record(created) || typeof created.query !== 'function') {
        throw new Error(`Collection '${collection_name}' not found and could not be created.`);
      }
      this.collection = created as ChromaVectorStore['collection'];
    }
  }

  similarity_search(query: string, k: number = 5, filters: RAGMetadataFilter | null = null): RAGDocument[] {
    const results = this.collection.query({
      query_texts: [query],
      n_results: k,
      where: equality_filter(filters),
      include: ['documents', 'metadatas', 'distances'],
    });
    return this._format_results(results);
  }

  vector_search(query_vector: number[], k: number = 5, filters: RAGMetadataFilter | null = null): RAGDocument[] {
    const results = this.collection.query({
      query_embeddings: [query_vector],
      n_results: k,
      where: equality_filter(filters),
      include: ['documents', 'metadatas', 'distances'],
    });
    return this._format_results(results);
  }

  get_collection_info(): Record<string, unknown> {
    const count = this.collection.count?.() ?? 0;
    const sample = count > 0 ? this.collection.peek?.({ limit: 1 }) : null;
    const embeddings = is_record(sample) && Array.isArray(sample.embeddings) ? sample.embeddings : [];
    const first_embedding = Array.isArray(embeddings[0]) ? embeddings[0] : [];
    return { name: this.collection_name, document_count: count, dimension: first_embedding.length, vector_store_type: 'chromadb' };
  }

  override supports_metadata_filtering(): boolean {
    return true;
  }

  _format_results(results: Record<string, unknown>): RAGDocument[] {
    const documents = Array.isArray(results.documents) && Array.isArray(results.documents[0]) ? results.documents[0] : [];
    const metadatas = Array.isArray(results.metadatas) && Array.isArray(results.metadatas[0]) ? results.metadatas[0] : [];
    const distances = Array.isArray(results.distances) && Array.isArray(results.distances[0]) ? results.distances[0] : [];
    return documents.map((doc, idx) => ({
      content: string_value(doc),
      metadata: metadata(metadatas[idx]),
      score: Math.max(0, 1 - numeric(distances[idx])),
    }));
  }
}

export class LanceDBVectorStore extends VectorStoreInterface {
  readonly db: { open_table?: (name: string) => LanceTable; create_table?: (name: string, args: { data: Array<Record<string, unknown>> }) => LanceTable };
  readonly table_name: string;
  readonly embedding_function: EmbeddingFunction | null;
  table: LanceTable | null;

  constructor(db: LanceDBVectorStore['db'], table_name: string, embedding_function: EmbeddingFunction | null = null) {
    super();
    this.db = db;
    this.table_name = table_name;
    this.embedding_function = embedding_function;
    try {
      this.table = db.open_table?.(table_name) ?? null;
    } catch {
      this.table = null;
    }
  }

  similarity_search(query: string, k: number = 5, filters: RAGMetadataFilter | null = null): RAGDocument[] {
    return this.vector_search(require_embedding(this.embedding_function, query), k, filters);
  }

  vector_search(query_vector: number[], k: number = 5, filters: RAGMetadataFilter | null = null): RAGDocument[] {
    if (this.table === null) return [];
    let builder = this.table.search(query_vector).limit(k);
    if (filters) builder = builder.where(this._convert_filters(filters));
    return this._format_results(builder.to_pandas());
  }

  add_documents(documents: Array<Record<string, unknown>>, embeddings: number[][], ids: string[] | null = null): string[] {
    const rows = prepare_rows(documents, embeddings, ids);
    if (this.table === null) {
      if (this.db.create_table === undefined) throw new Error('LanceDB create_table is unavailable.');
      this.table = this.db.create_table(this.table_name, { data: rows });
    } else {
      this.table.add(rows, { mode: 'append' });
    }
    return rows.map((row) => string_value(row.id));
  }

  delete_documents(ids: string[]): boolean {
    if (this.table === null) return false;
    this.table.delete(ids.length === 1 ? { id: ids[0] } : { id: { $in: ids } });
    return true;
  }

  get_collection_info(): Record<string, unknown> {
    const row_count = this.table?.count_rows?.() ?? 0;
    return { name: this.table_name, document_count: row_count, dimension: 0, vector_store_type: 'lancedb', schema: String(this.table?.schema ?? '') };
  }

  override supports_hybrid_search(): boolean {
    return true;
  }

  override hybrid_search(query: string, k: number = 5, _alpha: number = 0.5, filters: RAGMetadataFilter | null = null): RAGDocument[] {
    return this.similarity_search(query, k, filters);
  }

  _convert_filters(filters: RAGMetadataFilter): Record<string, unknown> {
    return filters;
  }

  _format_results(results: unknown): RAGDocument[] {
    const rows = Array.isArray(results) ? results : is_record(results) && Array.isArray(results.rows) ? results.rows : [];
    return rows.map(row_to_document);
  }
}

export type LanceTable = {
  search: (query_vector: number[]) => { limit: (k: number) => LanceQuery };
  add: (rows: Array<Record<string, unknown>>, options?: Record<string, unknown>) => void;
  delete: (filter: unknown) => void;
  count_rows?: () => number;
  schema?: unknown;
};
export type LanceQuery = {
  where: (filter: unknown) => LanceQuery;
  to_pandas: () => unknown;
};

export class MilvusVectorStore extends VectorStoreInterface {
  readonly client: {
    has_collection?: (name: string) => boolean;
    load_collection?: (name: string) => void;
    search: (args: Record<string, unknown>) => unknown;
    insert?: (args: Record<string, unknown>) => Record<string, unknown>;
    delete?: (args: Record<string, unknown>) => Record<string, unknown>;
    describe_collection?: (name: string) => Record<string, unknown>;
    get_collection_stats?: (name: string) => Record<string, unknown>;
  };
  readonly collection_name: string;
  readonly embedding_function: EmbeddingFunction | null;

  constructor(client: MilvusVectorStore['client'], collection_name: string, embedding_function: EmbeddingFunction | null = null) {
    super();
    if (client.has_collection?.(collection_name) === false) {
      throw new Error(`Collection '${collection_name}' not found. Please create the collection first.`);
    }
    this.client = client;
    this.collection_name = collection_name;
    this.embedding_function = embedding_function;
    try {
      client.load_collection?.(collection_name);
    } catch {
      // Already loaded is acceptable.
    }
  }

  similarity_search(query: string, k: number = 5, filters: RAGMetadataFilter | null = null): RAGDocument[] {
    return this.vector_search(require_embedding(this.embedding_function, query), k, filters);
  }

  vector_search(query_vector: number[], k: number = 5, filters: RAGMetadataFilter | null = null): RAGDocument[] {
    const results = this.client.search({
      collection_name: this.collection_name,
      data: [query_vector],
      limit: k,
      filter: filters ? this._convert_filters(filters) : null,
      output_fields: ['*'],
    });
    return this._format_results(results);
  }

  add_documents(documents: Array<Record<string, unknown>>, embeddings: number[][], ids: string[] | null = null): string[] {
    const rows = prepare_rows(documents, embeddings, ids);
    const result = this.client.insert?.({ collection_name: this.collection_name, data: rows }) ?? {};
    const inserted = result.ids;
    return Array.isArray(inserted) ? inserted.map(String) : rows.map((row) => string_value(row.id));
  }

  delete_documents(ids: string[]): boolean {
    const result = this.client.delete?.({ collection_name: this.collection_name, ids }) ?? {};
    return numeric(result.delete_count) > 0;
  }

  get_collection_info(): Record<string, unknown> {
    const description = this.client.describe_collection?.(this.collection_name) ?? {};
    const stats = this.client.get_collection_stats?.(this.collection_name) ?? {};
    const fields = Array.isArray(description.fields) ? description.fields : [];
    const vector_field = fields.find((field) => is_record(field) && field.type === 'FloatVector') as Record<string, unknown> | undefined;
    return {
      name: this.collection_name,
      document_count: numeric(stats.row_count),
      dimension: is_record(vector_field?.params) ? numeric(vector_field.params.dim) : 0,
      vector_store_type: 'milvus',
      vector_field: string_value(vector_field?.name, 'vector'),
      schema: description,
    };
  }

  override supports_hybrid_search(): boolean {
    return true;
  }

  override hybrid_search(query: string, k: number = 5, _alpha: number = 0.5, filters: RAGMetadataFilter | null = null): RAGDocument[] {
    return this.similarity_search(query, k, filters);
  }

  _convert_filters(filters: RAGMetadataFilter): string {
    return Object.entries(filters).map(([key, value]) => `${key} == ${JSON.stringify(value)}`).join(' && ');
  }

  _format_results(results: unknown): RAGDocument[] {
    const hits = Array.isArray(results) && Array.isArray(results[0]) ? results[0] : [];
    return hits.map(row_to_document);
  }
}

export class QdrantVectorStore extends VectorStoreInterface {
  readonly client: {
    get_collection?: (name: string) => unknown;
    query_points: (args: Record<string, unknown>) => unknown;
    upsert?: (args: Record<string, unknown>) => { status?: { name?: string } };
    delete?: (args: Record<string, unknown>) => { status?: { name?: string } };
  };
  readonly collection_name: string;
  readonly embedding_function: EmbeddingFunction | null;

  constructor(client: QdrantVectorStore['client'], collection_name: string, embedding_function: EmbeddingFunction | null = null) {
    super();
    try {
      client.get_collection?.(collection_name);
    } catch (error) {
      throw new Error(`Collection '${collection_name}' not found. Please create the collection first. Error: ${String(error)}`);
    }
    this.client = client;
    this.collection_name = collection_name;
    this.embedding_function = embedding_function;
  }

  similarity_search(query: string, k: number = 5, filters: RAGMetadataFilter | null = null): RAGDocument[] {
    return this.vector_search(require_embedding(this.embedding_function, query), k, filters);
  }

  vector_search(query_vector: number[], k: number = 5, filters: RAGMetadataFilter | null = null): RAGDocument[] {
    return this._format_results(this.client.query_points({
      collection_name: this.collection_name,
      query: query_vector,
      query_filter: filters ? this._convert_filters(filters) : null,
      limit: k,
      with_payload: true,
      with_vectors: false,
      score_threshold: null,
    }));
  }

  add_documents(documents: Array<Record<string, unknown>>, embeddings: number[][], ids: string[] | null = null): string[] {
    const rows = prepare_rows(documents, embeddings, ids);
    const points = rows.map((row, idx) => ({ id: idx, vector: row.vector, payload: { ...row, original_id: row.id } }));
    const result = this.client.upsert?.({ collection_name: this.collection_name, points, wait: true });
    if (result?.status?.name !== undefined && result.status.name !== 'COMPLETED') {
      throw new Error(`Upsert operation failed: ${result.status.name}`);
    }
    return rows.map((row) => string_value(row.id));
  }

  delete_documents(ids: string[]): boolean {
    const result = this.client.delete?.({ collection_name: this.collection_name, points_selector: { filter: this._convert_filters({ original_id: { $in: ids } }) }, wait: true });
    return result?.status?.name === undefined || result.status.name === 'COMPLETED';
  }

  get_collection_info(): Record<string, unknown> {
    const info = this.client.get_collection?.(this.collection_name);
    const record = is_record(info) ? info : {};
    const config = is_record(record.config) ? record.config : {};
    const params = is_record(config.params) ? config.params : {};
    const vectors = is_record(params.vectors) ? params.vectors : {};
    return {
      name: this.collection_name,
      document_count: numeric(record.points_count),
      dimension: numeric(vectors.size),
      vector_store_type: 'qdrant',
      distance_metric: string_value(is_record(vectors.distance) ? vectors.distance.name : vectors.distance, 'unknown').toLowerCase(),
      status: string_value(is_record(record.status) ? record.status.name : record.status, 'unknown').toLowerCase(),
    };
  }

  override supports_hybrid_search(): boolean {
    return true;
  }

  override hybrid_search(query: string, k: number = 5, _alpha: number = 0.5, filters: RAGMetadataFilter | null = null): RAGDocument[] {
    return this.similarity_search(query, k, filters);
  }

  _convert_filters(filters: RAGMetadataFilter): Record<string, unknown> {
    return { must: Object.entries(filters).map(([key, value]) => ({ key, match: is_record(value) ? value : { value } })) };
  }

  _format_results(results: unknown): RAGDocument[] {
    const points = is_record(results) && Array.isArray(results.points) ? results.points : Array.isArray(results) ? results : [];
    return points.map(row_to_document);
  }
}

export class WeaviateVectorStore extends VectorStoreInterface {
  readonly client: { collections?: { get: (name: string) => WeaviateCollection } };
  readonly collection_name: string;
  readonly embedding_function: EmbeddingFunction | null;
  readonly collection: WeaviateCollection;

  constructor(client: WeaviateVectorStore['client'], collection_name: string, embedding_function: EmbeddingFunction | null = null) {
    super();
    this.client = client;
    this.collection_name = collection_name;
    this.embedding_function = embedding_function;
    try {
      const collection = client.collections?.get(collection_name);
      if (collection === undefined) throw new Error('missing collection');
      this.collection = collection;
    } catch (error) {
      throw new Error(`Collection '${collection_name}' not found. Please create the collection first. Error: ${String(error)}`);
    }
  }

  similarity_search(query: string, k: number = 5, filters: RAGMetadataFilter | null = null): RAGDocument[] {
    return this.vector_search(require_embedding(this.embedding_function, query), k, filters);
  }

  vector_search(query_vector: number[], k: number = 5, filters: RAGMetadataFilter | null = null): RAGDocument[] {
    const near = this.collection.query.near_vector({ near_vector: query_vector, limit: k, filters: this._convert_filters(filters) });
    return this._format_results(near);
  }

  override hybrid_search(query: string, k: number = 5, alpha: number = 0.5, filters: RAGMetadataFilter | null = null): RAGDocument[] {
    return this._format_results(this.collection.query.hybrid({ query, alpha, limit: k, where: this._convert_filters(filters), return_properties: ['content', '*'] }));
  }

  get_collection_info(): Record<string, unknown> {
    const count_result = this.collection.aggregate?.over_all?.({ total_count: true });
    const config = this.collection.config?.get?.() ?? {};
    return {
      name: this.collection_name,
      document_count: numeric(is_record(count_result) ? count_result.total_count : 0),
      dimension: 0,
      vector_store_type: 'weaviate',
      supports_hybrid_search: this.supports_hybrid_search(),
      vectorizer: is_record(config) ? config.vectorizer_config : undefined,
      properties: is_record(config) && Array.isArray(config.properties) ? config.properties.map((prop) => is_record(prop) ? prop.name : prop) : [],
    };
  }

  override supports_hybrid_search(): boolean {
    return typeof this.collection.query.hybrid === 'function';
  }

  override supports_metadata_filtering(): boolean {
    return true;
  }

  _convert_filters(filters: RAGMetadataFilter | null): Record<string, unknown> | null {
    return filters ? { ...filters } : null;
  }

  _format_results(results: unknown): RAGDocument[] {
    const objects = is_record(results) && Array.isArray(results.objects) ? results.objects : Array.isArray(results) ? results : [];
    return objects.map(row_to_document);
  }
}

export type WeaviateCollection = {
  query: {
    near_vector: (args: Record<string, unknown>) => unknown;
    hybrid: (args: Record<string, unknown>) => unknown;
  };
  aggregate?: { over_all?: (args: Record<string, unknown>) => unknown };
  config?: { get?: () => unknown };
};

function prepare_rows(documents: Array<Record<string, unknown>>, embeddings: number[][], ids: string[] | null): Array<Record<string, unknown>> {
  if (documents.length !== embeddings.length) {
    throw new Error('Number of documents must match number of embeddings');
  }
  if (ids !== null && ids.length !== documents.length) {
    throw new Error('Number of IDs must match number of documents');
  }
  return documents.map((doc, idx) => ({
    id: ids?.[idx] ?? `doc_${idx}`,
    vector: embeddings[idx],
    ...doc,
  }));
}

function row_to_document(row: unknown): RAGDocument {
  const record = is_record(row) ? row : {};
  const payload = metadata(record.payload);
  const entity = metadata(record.entity);
  const properties = metadata(record.properties);
  const source = Object.keys(payload).length > 0 ? payload : Object.keys(entity).length > 0 ? entity : Object.keys(properties).length > 0 ? properties : record;
  const content = string_value(source.content ?? source.text ?? source.document);
  const metadata_record = metadata(source.metadata);
  if (Object.keys(metadata_record).length === 0) {
    for (const [key, value] of Object.entries(source)) {
      if (!['content', 'text', 'document', 'vector'].includes(key)) {
        metadata_record[key] = value;
      }
    }
  }
  const score = record.score ?? record.distance ?? record._distance ?? source.score ?? (is_record(record.metadata) ? record.metadata.score : undefined);
  const doc: RAGDocument = {
    content,
    metadata: metadata_record,
  };
  if (typeof score === 'number') doc.score = score;
  return doc;
}
