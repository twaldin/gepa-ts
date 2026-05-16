import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { GEPAEngine, GEPAState, ListDataLoader, StagedDataLoader, result_from_dict } from '../../src/core/index.js';
import type { GEPAAdapter } from '../../src/core/adapter.js';
import { ListDataLoader as DirectListDataLoader } from '../../src/core/data_loader.js';
import { GEPAState as DirectGEPAState } from '../../src/core/state.js';
import { ReflectiveMutationProposer } from '../../src/proposer/index.js';
import type { CandidateProposal } from '../../src/proposer/base.js';
import { MergeProposer } from '../../src/proposer/merge.js';
import { ReflectiveMutationProposer as DirectReflectiveMutationProposer } from '../../src/proposer/reflective_mutation/reflective_mutation.js';
import type { LanguageModel } from '../../src/proposer/reflective_mutation/base.js';
import { ParetoCandidateSelector, RoundRobinSampleEvaluationPolicy } from '../../src/strategies/index.js';
import { StrictImprovementAcceptance } from '../../src/acceptance.js';
import { EpochShuffledBatchSampler } from '../../src/batch_sampler.js';
import { ExecutionMode, MaxMetricCallsStopper, execute_code, idxmax } from '../../src/utils/index.js';
import { idxmax as gepa_utils_idxmax } from '../../src/gepa_utils.js';
import { ThreadLocalStreamCapture } from '../../src/utils/stdio_capture.js';
import { AnyMathsAdapter, ChromaVectorStore, ConfidenceAdapter, DSPyInstructionProposalSignature, DSPyProgramProposalSignature, DefaultAdapter, DspyAdapter, DspyFullProgramAdapter, GenericRAGAdapter, MCPAdapter, SSEMCPClient, StdioMCPClient, StreamableHTTPMCPClient, TerminusAdapter } from '../../src/adapters/index.js';
import { OptimizeAnythingAdapter as DirectOptimizeAnythingAdapter } from '../../src/adapters/optimize_anything_adapter/optimize_anything_adapter.js';
import { DefaultAdapter as DirectDefaultAdapter } from '../../src/adapters/default_adapter/default_adapter.js';
import { ConfidenceAdapter as DirectConfidenceAdapter } from '../../src/adapters/confidence_adapter/confidence_adapter.js';
import { LinearBlendScoring } from '../../src/adapters/confidence_adapter/scoring.js';
import { RAGEvaluationMetrics } from '../../src/adapters/generic_rag_adapter/evaluation_metrics.js';
import { GenericRAGAdapter as DirectGenericRAGAdapter } from '../../src/adapters/generic_rag_adapter/generic_rag_adapter.js';
import { RAGPipeline } from '../../src/adapters/generic_rag_adapter/rag_pipeline.js';
import { VectorStoreInterface } from '../../src/adapters/generic_rag_adapter/vector_store_interface.js';
import { AnyMathsAdapter as DirectAnyMathsAdapter } from '../../src/adapters/anymaths_adapter/anymaths_adapter.js';
import { TerminusAdapter as DirectTerminusAdapter } from '../../src/adapters/terminal_bench_adapter/terminal_bench_adapter.js';
import { MCPAdapter as DirectMCPAdapter } from '../../src/adapters/mcp_adapter/mcp_adapter.js';
import { create_mcp_client } from '../../src/adapters/mcp_adapter/mcp_client.js';
import { DspyAdapter as DirectDspyAdapter } from '../../src/adapters/dspy_adapter/dspy_adapter.js';
import { DspyAdapter as DirectDspyFullProgramAdapter } from '../../src/adapters/dspy_full_program_adapter/full_program_adapter.js';
import { candidate_tree_dot_from_data } from '../../src/visualization.js';
import { LM, TrackingLM, make_litellm_lm } from '../../src/lm.js';
import { Image } from '../../src/image.js';
import { ExperimentTracker, create_experiment_tracker } from '../../src/logging/index.js';
import { log_detailed_metrics_after_discovering_new_program } from '../../src/logging/utils.js';
import { log_detailed_metrics_after_discovering_new_program as root_log_detailed_metrics } from '../../src/index.js';

describe('package subpath exports', () => {
  it('exposes upstream-shaped native module groups from source entrypoints', async () => {
    expect(GEPAEngine).toBeTypeOf('function');
    expect(GEPAState).toBeTypeOf('function');
    expect(result_from_dict({ candidates: [], parents: [], val_aggregate_scores: [], discovery_eval_counts: [] }).num_candidates).toBe(0);
    expect(new ListDataLoader(['a']).fetch([0])).toEqual(['a']);
    expect(new DirectListDataLoader(['a']).fetch([0])).toEqual(['a']);
    expect(new StagedDataLoader(['a'], [[1, ['b']]]).unlock_next_stage()).toBe(true);
    expect(DirectGEPAState).toBe(GEPAState);
    expect(ReflectiveMutationProposer).toBeTypeOf('function');
    expect(DirectReflectiveMutationProposer).toBe(ReflectiveMutationProposer);
    expect(MergeProposer).toBeTypeOf('function');
    expect(ParetoCandidateSelector).toBeTypeOf('function');
    expect(StrictImprovementAcceptance).toBeTypeOf('function');
    expect(EpochShuffledBatchSampler).toBeTypeOf('function');
    expect(RoundRobinSampleEvaluationPolicy).toBeTypeOf('function');
    expect(new MaxMetricCallsStopper(1)({ total_num_evals: 1 })).toBe(true);
    expect(idxmax([0.1, 0.9])).toBe(1);
    expect(gepa_utils_idxmax([0.1, 0.9])).toBe(1);
    expect(execute_code('const value = 1;', { mode: ExecutionMode.IN_PROCESS }).success).toBe(true);
    expect(new ThreadLocalStreamCapture()).toBeTypeOf('object');
    expect(DefaultAdapter).toBeTypeOf('function');
    expect(DirectOptimizeAnythingAdapter).toBeTypeOf('function');
    expect(DirectDefaultAdapter).toBe(DefaultAdapter);
    expect(GenericRAGAdapter).toBeTypeOf('function');
    expect(DirectGenericRAGAdapter).toBe(GenericRAGAdapter);
    expect(ChromaVectorStore).toBeTypeOf('function');
    expect(ConfidenceAdapter).toBeTypeOf('function');
    expect(DirectConfidenceAdapter).toBe(ConfidenceAdapter);
    expect(AnyMathsAdapter).toBeTypeOf('function');
    expect(DirectAnyMathsAdapter).toBe(AnyMathsAdapter);
    expect(TerminusAdapter).toBeTypeOf('function');
    expect(DirectTerminusAdapter).toBe(TerminusAdapter);
    expect(MCPAdapter).toBeTypeOf('function');
    expect(DirectMCPAdapter).toBe(MCPAdapter);
    expect(SSEMCPClient).toBeTypeOf('function');
    expect(StdioMCPClient).toBeTypeOf('function');
    expect(StreamableHTTPMCPClient).toBeTypeOf('function');
    expect(create_mcp_client).toBeTypeOf('function');
    expect(DspyAdapter).toBeTypeOf('function');
    expect(DirectDspyAdapter).toBe(DspyAdapter);
    expect(DSPyInstructionProposalSignature).toBeTypeOf('function');
    expect(DspyFullProgramAdapter).toBeTypeOf('function');
    expect(DirectDspyFullProgramAdapter).toBe(DspyFullProgramAdapter);
    expect(DSPyProgramProposalSignature).toBeTypeOf('function');
    expect(LinearBlendScoring).toBeTypeOf('function');
    expect(RAGEvaluationMetrics).toBeTypeOf('function');
    expect(RAGPipeline).toBeTypeOf('function');
    expect(VectorStoreInterface).toBeTypeOf('function');
    expect(candidate_tree_dot_from_data([], [], [], {})).toContain('digraph G');
    expect(new LM('model', { completion: () => 'x' })).toBeTypeOf('function');
    expect(make_litellm_lm('model', { completion: () => 'x' })).toBeTypeOf('function');
    expect(new TrackingLM(async () => 'x')).toBeTypeOf('function');
    expect(new Image({ url: 'https://example.test/a.png' }).to_openai_content_part().type).toBe('image_url');
    expect(create_experiment_tracker()).toBeInstanceOf(ExperimentTracker);
    expect(log_detailed_metrics_after_discovering_new_program).toBeTypeOf('function');
    expect(root_log_detailed_metrics).toBe(log_detailed_metrics_after_discovering_new_program);

    const adapter: GEPAAdapter = {
      evaluate: async () => ({ outputs: [], scores: [] }),
      make_reflective_dataset: () => ({}),
    };
    expect(adapter).toBeTypeOf('object');
    const proposal: CandidateProposal = { candidate: {}, parent_program_ids: [] };
    const lm: LanguageModel = async () => 'x';
    expect(proposal.parent_program_ids).toEqual([]);
    await expect(lm('prompt')).resolves.toBe('x');
  });

  it('declares built package exports for the native module groups', () => {
    const pkg = JSON.parse(readFileSync('package.json', 'utf8')) as {
      exports: Record<string, { types: string; import: string; require: string }>;
    };

    for (const subpath of [
      '.',
      './core',
      './core/*',
      './proposer',
      './proposer/*',
      './proposer/merge',
      './proposer/reflective_mutation',
      './proposer/reflective_mutation/*',
      './strategies',
      './strategies/*',
      './utils',
      './utils/*',
      './utils/code_execution',
      './utils/stdio_capture',
      './gepa_utils',
      './lm',
      './image',
      './visualization',
      './logging',
      './logging/*',
      './adapters',
      './adapters/optimize_anything_adapter',
      './adapters/optimize_anything_adapter/*',
      './adapters/default_adapter',
      './adapters/default_adapter/*',
      './adapters/generic_rag_adapter',
      './adapters/generic_rag_adapter/*',
      './adapters/generic_rag_adapter/vector_stores',
      './adapters/generic_rag_adapter/vector_stores/*',
      './adapters/confidence_adapter',
      './adapters/confidence_adapter/*',
      './adapters/anymaths_adapter',
      './adapters/anymaths_adapter/*',
      './adapters/terminal_bench_adapter',
      './adapters/terminal_bench_adapter/*',
      './adapters/mcp_adapter',
      './adapters/mcp_adapter/*',
      './adapters/dspy_adapter',
      './adapters/dspy_adapter/*',
      './adapters/dspy_full_program_adapter',
      './adapters/dspy_full_program_adapter/*',
    ]) {
      expect(pkg.exports[subpath]?.types).toMatch(/^\.\/dist\//);
      expect(pkg.exports[subpath]?.import).toMatch(/^\.\/dist\//);
      expect(pkg.exports[subpath]?.require).toMatch(/^\.\/dist\//);
    }
  });
});
