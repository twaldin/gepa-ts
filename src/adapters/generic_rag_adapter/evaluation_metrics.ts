import type { RAGDocument } from './vector_store_interface.js';

export type RetrievalMetrics = {
  retrieval_precision: number;
  retrieval_recall: number;
  retrieval_f1: number;
  retrieval_mrr: number;
};

export type GenerationMetrics = {
  exact_match: number;
  token_f1: number;
  bleu_score: number;
  answer_relevance: number;
  faithfulness: number;
  answer_confidence: number;
};

function empty_retrieval_metrics(): RetrievalMetrics {
  return {
    retrieval_precision: 0.0,
    retrieval_recall: 0.0,
    retrieval_f1: 0.0,
    retrieval_mrr: 0.0,
  };
}

export class RAGEvaluationMetrics {
  evaluate_retrieval(retrieved_docs: RAGDocument[], relevant_doc_ids: string[]): RetrievalMetrics {
    if (retrieved_docs.length === 0 || relevant_doc_ids.length === 0) {
      return empty_retrieval_metrics();
    }

    const retrieved_ids: string[] = [];
    for (const doc of retrieved_docs) {
      const metadata = doc.metadata ?? {};
      const doc_id = metadata['doc_id'] ?? metadata['id'];
      if (doc_id !== undefined && doc_id !== null && doc_id !== '') {
        retrieved_ids.push(String(doc_id));
      }
    }

    const relevant_set = new Set(relevant_doc_ids);
    const retrieved_set = new Set(retrieved_ids);
    const intersection_size = [...relevant_set].filter((id) => retrieved_set.has(id)).length;

    const precision = retrieved_set.size === 0 ? 0.0 : intersection_size / retrieved_set.size;
    const recall = relevant_set.size === 0 ? 0.0 : intersection_size / relevant_set.size;
    const f1 = precision + recall === 0 ? 0.0 : 2 * (precision * recall) / (precision + recall);

    let mrr = 0.0;
    for (let i = 0; i < retrieved_ids.length; i += 1) {
      const retrieved_id = retrieved_ids[i];
      if (retrieved_id !== undefined && relevant_set.has(retrieved_id)) {
        mrr = 1.0 / (i + 1);
        break;
      }
    }

    return {
      retrieval_precision: precision,
      retrieval_recall: recall,
      retrieval_f1: f1,
      retrieval_mrr: mrr,
    };
  }

  evaluate_generation(generated_answer: string, ground_truth: string, context: string): GenerationMetrics {
    const exact_match = this._exact_match(generated_answer, ground_truth) ? 1.0 : 0.0;
    const token_f1 = this._token_f1(generated_answer, ground_truth);
    const bleu_score = this._simple_bleu(generated_answer, ground_truth);
    const answer_relevance = this._answer_relevance(generated_answer, context);
    const faithfulness = this._faithfulness_score(generated_answer, context);

    return {
      exact_match,
      token_f1,
      bleu_score,
      answer_relevance,
      faithfulness,
      answer_confidence: (token_f1 + answer_relevance + faithfulness) / 3.0,
    };
  }

  combined_rag_score(
    retrieval_metrics: Partial<RetrievalMetrics>,
    generation_metrics: Partial<GenerationMetrics>,
    retrieval_weight: number = 0.3,
    generation_weight: number = 0.7,
  ): number {
    const retrieval_score = retrieval_metrics.retrieval_f1 ?? 0.0;
    const generation_score =
      (generation_metrics.token_f1 ?? 0.0) * 0.4 +
      (generation_metrics.answer_relevance ?? 0.0) * 0.3 +
      (generation_metrics.faithfulness ?? 0.0) * 0.3;

    return retrieval_weight * retrieval_score + generation_weight * generation_score;
  }

  _exact_match(prediction: string, ground_truth: string): boolean {
    return prediction.trim().toLowerCase() === ground_truth.trim().toLowerCase();
  }

  _token_f1(prediction: string, ground_truth: string): number {
    const pred_tokens = new Set(this._normalize_text(prediction).split(' ').filter(Boolean));
    const truth_tokens = new Set(this._normalize_text(ground_truth).split(' ').filter(Boolean));

    if (pred_tokens.size === 0 && truth_tokens.size === 0) return 1.0;
    if (pred_tokens.size === 0 || truth_tokens.size === 0) return 0.0;

    const intersection_size = [...pred_tokens].filter((token) => truth_tokens.has(token)).length;
    const precision = intersection_size / pred_tokens.size;
    const recall = intersection_size / truth_tokens.size;

    return precision + recall === 0 ? 0.0 : 2 * (precision * recall) / (precision + recall);
  }

  _simple_bleu(prediction: string, ground_truth: string, n: number = 2): number {
    const pred_words = this._normalize_text(prediction).split(' ').filter(Boolean);
    const truth_words = this._normalize_text(ground_truth).split(' ').filter(Boolean);

    if (pred_words.length < n || truth_words.length < n) {
      return this._token_f1(prediction, ground_truth);
    }

    const pred_ngrams = new Set<string>();
    const truth_ngrams = new Set<string>();
    for (let i = 0; i <= pred_words.length - n; i += 1) {
      pred_ngrams.add(pred_words.slice(i, i + n).join('\u0000'));
    }
    for (let i = 0; i <= truth_words.length - n; i += 1) {
      truth_ngrams.add(truth_words.slice(i, i + n).join('\u0000'));
    }

    if (pred_ngrams.size === 0 || truth_ngrams.size === 0) return 0.0;
    const intersection_size = [...pred_ngrams].filter((ngram) => truth_ngrams.has(ngram)).length;
    return intersection_size / pred_ngrams.size;
  }

  _answer_relevance(answer: string, context: string): number {
    const answer_words = new Set(this._normalize_text(answer).split(' ').filter(Boolean));
    const context_words = new Set(this._normalize_text(context).split(' ').filter(Boolean));

    if (answer_words.size === 0) return 0.0;
    const overlap_size = [...answer_words].filter((word) => context_words.has(word)).length;
    return overlap_size / answer_words.size;
  }

  _faithfulness_score(answer: string, context: string): number {
    const answer_phrases = this._extract_phrases(answer);
    const context_phrases = this._extract_phrases(context);

    if (answer_phrases.size === 0) return 1.0;
    const supported_size = [...answer_phrases].filter((phrase) => context_phrases.has(phrase)).length;
    return supported_size / answer_phrases.size;
  }

  _extract_phrases(text: string, min_length: number = 2): Set<string> {
    const words = this._normalize_text(text).split(' ').filter(Boolean);
    const phrases = new Set<string>();

    for (const word of words) {
      if (word.length > 3) phrases.add(word);
    }

    const max_n = Math.min(4, words.length + 1);
    for (let n = min_length; n < max_n; n += 1) {
      for (let i = 0; i <= words.length - n; i += 1) {
        const phrase = words.slice(i, i + n).join(' ');
        if (phrase.length > 5) phrases.add(phrase);
      }
    }

    return phrases;
  }

  _normalize_text(text: string): string {
    return text
      .toLowerCase()
      .trim()
      .replace(/[^\w\s]/g, ' ')
      .replace(/\s+/g, ' ');
  }
}
