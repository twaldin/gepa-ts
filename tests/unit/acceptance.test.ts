import { describe, expect, test } from "vitest";
import { ImprovementOrEqualAcceptance, StrictImprovementAcceptance } from "../../src/acceptance";

const base_proposal = {
  candidate: {},
  parent_program_ids: [0],
};

describe("StrictImprovementAcceptance", () => {
  test("accepts only strict improvements", () => {
    const acceptance = new StrictImprovementAcceptance();
    expect(
      acceptance.should_accept({ ...base_proposal, subsample_scores_before: [1], subsample_scores_after: [2] }, { total_num_evals: 0 }),
    ).toBe(true);
    expect(
      acceptance.should_accept({ ...base_proposal, subsample_scores_before: [1], subsample_scores_after: [1] }, { total_num_evals: 0 }),
    ).toBe(false);
    expect(
      acceptance.should_accept({ ...base_proposal, subsample_scores_before: [0], subsample_scores_after: [0] }, { total_num_evals: 0 }),
    ).toBe(false);
  });

  test("treats missing arrays as zero", () => {
    const acceptance = new StrictImprovementAcceptance();
    expect(acceptance.should_accept({ ...base_proposal }, { total_num_evals: 0 })).toBe(false);
  });
});

describe("ImprovementOrEqualAcceptance", () => {
  test("accepts improvements and ties", () => {
    const acceptance = new ImprovementOrEqualAcceptance();
    expect(
      acceptance.should_accept({ ...base_proposal, subsample_scores_before: [1], subsample_scores_after: [1] }, { total_num_evals: 0 }),
    ).toBe(true);
    expect(
      acceptance.should_accept({ ...base_proposal, subsample_scores_before: [1], subsample_scores_after: [0] }, { total_num_evals: 0 }),
    ).toBe(false);
  });

  test("treats missing arrays as zero", () => {
    const acceptance = new ImprovementOrEqualAcceptance();
    expect(acceptance.should_accept({ ...base_proposal }, { total_num_evals: 0 })).toBe(true);
  });
});
