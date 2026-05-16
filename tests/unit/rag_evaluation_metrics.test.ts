import { describe, expect, it } from 'vitest';
import { RAGEvaluationMetrics, type RAGDocument } from '../../src/index.js';

describe('RAGEvaluationMetrics', () => {
  it('evaluates perfect retrieval', () => {
    const metrics = new RAGEvaluationMetrics();
    const retrieved_docs: RAGDocument[] = [
      { content: '', metadata: { doc_id: 'doc1' } },
      { content: '', metadata: { doc_id: 'doc2' } },
      { content: '', metadata: { doc_id: 'doc3' } },
    ];

    expect(metrics.evaluate_retrieval(retrieved_docs, ['doc1', 'doc2', 'doc3'])).toEqual({
      retrieval_precision: 1,
      retrieval_recall: 1,
      retrieval_f1: 1,
      retrieval_mrr: 1,
    });
  });

  it('evaluates partial retrieval and id metadata aliases', () => {
    const metrics = new RAGEvaluationMetrics();
    const retrieved_docs: RAGDocument[] = [
      { content: '', metadata: { id: 'doc1' } },
      { content: '', metadata: { id: 'doc2' } },
      { content: '', metadata: { id: 'doc3' } },
      { content: '', metadata: { id: 'doc4' } },
    ];

    const result = metrics.evaluate_retrieval(retrieved_docs, ['doc1', 'doc3']);

    expect(result.retrieval_precision).toBe(0.5);
    expect(result.retrieval_recall).toBe(1);
    expect(result.retrieval_f1).toBeCloseTo(2 / 3);
    expect(result.retrieval_mrr).toBe(1);
  });

  it('returns zero retrieval metrics for empty inputs or no matches', () => {
    const metrics = new RAGEvaluationMetrics();

    expect(metrics.evaluate_retrieval([], ['doc1'])).toEqual({
      retrieval_precision: 0,
      retrieval_recall: 0,
      retrieval_f1: 0,
      retrieval_mrr: 0,
    });
    expect(metrics.evaluate_retrieval([{ content: '', metadata: { doc_id: 'doc1' } }], [])).toEqual({
      retrieval_precision: 0,
      retrieval_recall: 0,
      retrieval_f1: 0,
      retrieval_mrr: 0,
    });
    expect(metrics.evaluate_retrieval([{ content: '', metadata: { doc_id: 'doc1' } }], ['doc2']).retrieval_mrr).toBe(0);
  });

  it('evaluates generation with exact match and confidence metrics', () => {
    const metrics = new RAGEvaluationMetrics();

    const result = metrics.evaluate_generation(
      'Machine learning is a subset of AI.',
      'Machine learning is a subset of AI.',
      'Machine learning is a subset of artificial intelligence.',
    );

    expect(result.exact_match).toBe(1);
    expect(result.token_f1).toBe(1);
    expect(result.bleu_score).toBe(1);
    expect(result.answer_relevance).toBeGreaterThan(0);
    expect(result.faithfulness).toBeGreaterThanOrEqual(0);
    expect(result.answer_confidence).toBeGreaterThan(0);
  });

  it('matches upstream token and BLEU overlap behavior', () => {
    const metrics = new RAGEvaluationMetrics();

    expect(metrics._token_f1('machine learning algorithms', 'machine learning techniques')).toBeCloseTo(2 / 3);
    expect(metrics._token_f1('', '')).toBe(1);
    expect(metrics._token_f1('hello', '')).toBe(0);
    expect(metrics._simple_bleu('machine learning is amazing', 'machine learning is amazing')).toBe(1);
    expect(metrics._simple_bleu('completely different', 'totally unrelated')).toBe(0);
  });

  it('normalizes text and extracts faithfulness phrases', () => {
    const metrics = new RAGEvaluationMetrics();

    expect(metrics._exact_match('  Hello World  ', 'hello world')).toBe(true);
    expect(metrics._exact_match('hello\tworld', 'hello world')).toBe(false);
    expect(metrics._normalize_text("Hello! How are you? I'm fine, thanks.").trim()).toBe('hello how are you i m fine thanks');
    expect(metrics._extract_phrases('machine learning algorithms are powerful').size).toBeGreaterThan(0);
    expect(metrics._faithfulness_score('', 'machine learning context')).toBe(1);
  });

  it('combines retrieval and generation scores with upstream weights', () => {
    const metrics = new RAGEvaluationMetrics();

    const score = metrics.combined_rag_score(
      { retrieval_f1: 0.8 },
      { token_f1: 0.7, answer_relevance: 0.6, faithfulness: 0.9 },
    );

    const expected_generation_score = 0.7 * 0.4 + 0.6 * 0.3 + 0.9 * 0.3;
    expect(score).toBeCloseTo(0.3 * 0.8 + 0.7 * expected_generation_score);
  });
});
