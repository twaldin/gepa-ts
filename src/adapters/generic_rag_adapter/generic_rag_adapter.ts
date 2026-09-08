import type { Candidate, EvaluationBatch, GEPAAdapter } from '../../types.js';
import { RAGEvaluationMetrics } from './evaluation_metrics.js';
import { RAGPipeline, type RAGLLMClient, type RAGPipelineConfig } from './rag_pipeline.js';
import { type RAGDocument, VectorStoreInterface } from './vector_store_interface.js';

export type RAGDataInst = {
  query: string;
  ground_truth_answer: string;
  relevant_doc_ids: string[];
  metadata: Record<string, unknown>;
};

export type RAGTrajectory = {
  original_query: string;
  reformulated_query: string;
  retrieved_docs: RAGDocument[];
  synthesized_context: string;
  generated_answer: string;
  execution_metadata: Record<string, unknown>;
};

export type RAGOutput = {
  final_answer: string;
  confidence_score: number;
  retrieved_docs: RAGDocument[];
  total_tokens: number;
};

export type GenericRAGConfig = RAGPipelineConfig & {
  retrieval_weight?: number;
  generation_weight?: number;
};

function default_config(): GenericRAGConfig {
  return {
    retrieval_strategy: 'similarity',
    top_k: 5,
    retrieval_weight: 0.3,
    generation_weight: 0.7,
    hybrid_alpha: 0.5,
    filters: null,
  };
}

function error_message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export class GenericRAGAdapter implements GEPAAdapter<RAGDataInst, RAGTrajectory, RAGOutput> {
  readonly vector_store: VectorStoreInterface;
  readonly rag_pipeline: RAGPipeline;
  readonly evaluator: RAGEvaluationMetrics;
  readonly config: GenericRAGConfig;
  readonly failure_score: number;

  constructor({
    vector_store,
    llm_model,
    embedding_model = 'text-embedding-3-small',
    embedding_function = null,
    rag_config = null,
    failure_score = 0.0,
  }: {
    vector_store: VectorStoreInterface;
    llm_model: RAGLLMClient;
    embedding_model?: string;
    embedding_function?: ((text: string) => number[]) | null;
    rag_config?: GenericRAGConfig | null;
    failure_score?: number;
  }) {
    this.vector_store = vector_store;
    this.rag_pipeline = new RAGPipeline({
      vector_store,
      llm_client: llm_model,
      embedding_model,
      embedding_function,
    });
    this.evaluator = new RAGEvaluationMetrics();
    this.config = rag_config ?? default_config();
    this.failure_score = failure_score;
  }

  async evaluate(
    batch: RAGDataInst[],
    candidate: Candidate,
    capture_traces: boolean = false,
  ): Promise<EvaluationBatch<RAGTrajectory, RAGOutput>> {
    const outputs: RAGOutput[] = [];
    const scores: number[] = [];
    const trajectories: RAGTrajectory[] | undefined = capture_traces ? [] : undefined;

    for (const data_inst of batch) {
      try {
        const rag_result = await this.rag_pipeline.execute_rag(data_inst.query, candidate, this.config);
        const retrieval_metrics = this.evaluator.evaluate_retrieval(
          rag_result.retrieved_docs,
          data_inst.relevant_doc_ids,
        );
        const generation_metrics = this.evaluator.evaluate_generation(
          rag_result.generated_answer,
          data_inst.ground_truth_answer,
          rag_result.synthesized_context,
        );
        const overall_score = this.evaluator.combined_rag_score(
          retrieval_metrics,
          generation_metrics,
          this.config.retrieval_weight ?? 0.3,
          this.config.generation_weight ?? 0.7,
        );

        const output: RAGOutput = {
          final_answer: rag_result.generated_answer,
          confidence_score: generation_metrics.answer_confidence ?? 0.5,
          retrieved_docs: rag_result.retrieved_docs,
          total_tokens: rag_result.metadata.total_tokens,
        };

        outputs.push(output);
        scores.push(overall_score);

        if (trajectories !== undefined) {
          trajectories.push({
            original_query: rag_result.original_query,
            reformulated_query: rag_result.reformulated_query,
            retrieved_docs: rag_result.retrieved_docs,
            synthesized_context: rag_result.synthesized_context,
            generated_answer: rag_result.generated_answer,
            execution_metadata: {
              ...rag_result.metadata,
              retrieval_metrics,
              generation_metrics,
              overall_score,
            },
          });
        }
      } catch (error) {
        const message = error_message(error);
        outputs.push({
          final_answer: `Error: ${message}`,
          confidence_score: 0.0,
          retrieved_docs: [],
          total_tokens: 0,
        });
        scores.push(this.failure_score);

        if (trajectories !== undefined) {
          trajectories.push({
            original_query: data_inst.query,
            reformulated_query: data_inst.query,
            retrieved_docs: [],
            synthesized_context: '',
            generated_answer: `Error: ${message}`,
            execution_metadata: { error: message },
          });
        }
      }
    }

    return {
      outputs,
      scores,
      ...(trajectories !== undefined ? { trajectories } : {}),
      num_metric_calls: batch.length,
    };
  }

  make_reflective_dataset(
    candidate: Candidate,
    eval_batch: EvaluationBatch<RAGTrajectory, RAGOutput>,
    components_to_update: string[],
  ): Record<string, Array<Record<string, unknown>>> {
    const reflective_data: Record<string, Array<Record<string, unknown>>> = {};
    const trajectories = eval_batch.trajectories ?? [];

    for (const component of components_to_update) {
      const component_examples: Array<Record<string, unknown>> = [];

      for (let i = 0; i < trajectories.length; i += 1) {
        const trajectory = trajectories[i];
        const output = eval_batch.outputs[i];
        const score = eval_batch.scores[i];
        if (trajectory === undefined || output === undefined || score === undefined) continue;
        const example = this._create_component_example(component, trajectory, output, score, candidate);
        if (example !== null) component_examples.push(example);
      }

      if (component_examples.length > 0) {
        reflective_data[component] = component_examples;
      }
    }

    return reflective_data;
  }

  _create_component_example(
    component_name: string,
    trajectory: RAGTrajectory,
    output: RAGOutput,
    score: number,
    candidate: Candidate,
  ): Record<string, unknown> | null {
    if (component_name === 'query_reformulation') {
      return {
        Inputs: {
          original_query: trajectory.original_query,
          current_prompt: candidate[component_name] ?? '',
        },
        'Generated Outputs': trajectory.reformulated_query,
        Feedback: this._generate_query_reformulation_feedback(trajectory, score),
      };
    }

    if (component_name === 'context_synthesis') {
      return {
        Inputs: {
          query: trajectory.original_query,
          retrieved_docs: trajectory.retrieved_docs.map((doc) => doc.content),
          current_prompt: candidate[component_name] ?? '',
        },
        'Generated Outputs': trajectory.synthesized_context,
        Feedback: this._generate_context_synthesis_feedback(trajectory, score),
      };
    }

    if (component_name === 'answer_generation') {
      return {
        Inputs: {
          query: trajectory.original_query,
          context: trajectory.synthesized_context,
          current_prompt: candidate[component_name] ?? '',
        },
        'Generated Outputs': trajectory.generated_answer,
        Feedback: this._generate_answer_generation_feedback(trajectory, output, score),
      };
    }

    if (component_name === 'reranking_criteria') {
      return {
        Inputs: {
          query: trajectory.original_query,
          documents: trajectory.retrieved_docs.map((doc) => doc.content),
          current_criteria: candidate[component_name] ?? '',
        },
        'Generated Outputs': 'Document ranking applied',
        Feedback: this._generate_reranking_feedback(trajectory, score),
      };
    }

    return null;
  }

  _generate_query_reformulation_feedback(trajectory: RAGTrajectory, score: number): string {
    if (score > 0.7) {
      return `Good query reformulation. The reformulated query '${trajectory.reformulated_query}' helped retrieve relevant documents and generated a good answer.`;
    }
    return `The query reformulation from '${trajectory.original_query}' to '${trajectory.reformulated_query}' may not have improved retrieval. Consider making the reformulated query more specific or preserving key terms.`;
  }

  _generate_context_synthesis_feedback(_trajectory: RAGTrajectory, score: number): string {
    if (score > 0.7) {
      return 'Context synthesis worked well - the synthesized context effectively supported answer generation.';
    }
    return 'Context synthesis could be improved. The synthesized context may not have highlighted the most relevant information or may have been too verbose/concise.';
  }

  _generate_answer_generation_feedback(trajectory: RAGTrajectory, _output: RAGOutput, score: number): string {
    if (score > 0.7) {
      return `Good answer generation. The generated answer '${trajectory.generated_answer}' was accurate and well-supported by the context.`;
    }
    return `Answer generation needs improvement. The generated answer '${trajectory.generated_answer}' may not be fully accurate or well-supported by the provided context.`;
  }

  _generate_reranking_feedback(_trajectory: RAGTrajectory, score: number): string {
    if (score > 0.7) {
      return 'Document reranking appears to have helped surface more relevant documents for answer generation.';
    }
    return 'Document reranking may not have improved relevance. Consider adjusting the criteria to better prioritize documents that contain the answer.';
  }

  _default_config(): GenericRAGConfig {
    return default_config();
  }
}
