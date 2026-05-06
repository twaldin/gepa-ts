import type { AcceptanceCriterion, CandidateProposal, GEPAStateLike } from "./types";

function sum_scores(scores: number[] | undefined): number {
  if (!scores) {
    return 0;
  }
  return scores.reduce((acc, value) => acc + value, 0);
}

export class StrictImprovementAcceptance implements AcceptanceCriterion {
  should_accept(proposal: CandidateProposal, _state: GEPAStateLike): boolean {
    const old_sum = sum_scores(proposal.subsample_scores_before);
    const new_sum = sum_scores(proposal.subsample_scores_after);
    return new_sum > old_sum;
  }
}

export class ImprovementOrEqualAcceptance implements AcceptanceCriterion {
  should_accept(proposal: CandidateProposal, _state: GEPAStateLike): boolean {
    const old_sum = sum_scores(proposal.subsample_scores_before);
    const new_sum = sum_scores(proposal.subsample_scores_after);
    return new_sum >= old_sum;
  }
}
