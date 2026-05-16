export { ReflectiveMutationProposer } from '../proposer.js';
export type { ProposalContext, ProposalOutput } from '../proposer.js';
export {
  MergeProposer,
  does_triplet_have_desirable_predictors,
  filter_ancestors,
  find_common_ancestor_pair,
  sample_and_attempt_merge_programs_by_common_predictors,
} from './merge.js';
export type { AncestorLog, MergeAttempt, MergeDescription, MergesPerformed } from './merge.js';
export type {
  CandidateProposal,
  ProposeNewCandidate,
  SubsampleEvaluation,
} from '../types.js';
