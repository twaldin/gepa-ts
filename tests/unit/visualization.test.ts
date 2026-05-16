import { describe, expect, it } from 'vitest';
import {
  candidate_tree_dot_from_data,
  candidate_tree_html_from_data,
} from '../../src/index.js';
import { result_from_state } from '../../src/result.js';
import { initialize_gepa_state, ValsetEvaluation } from '../../src/state.js';

function sample_data() {
  const candidates = [
    { system_prompt: 'You are a helpful assistant.' },
    { system_prompt: 'You are an expert math tutor. Show step-by-step solutions.' },
    { system_prompt: 'You are a precise math solver. Always verify your answer.' },
  ];
  const parents = [[null], [0], [0]];
  const val_scores = [0.5, 0.7, 0.65];
  const pareto_front = {
    ex_0: new Set([1]),
    ex_1: new Set([1, 2]),
    ex_2: new Set([2]),
  };
  return { candidates, parents, val_scores, pareto_front };
}

describe('candidate tree visualization', () => {
  it('generates DOT with nodes, edges, and suppressed tooltips', () => {
    const { candidates, parents, val_scores, pareto_front } = sample_data();
    const dot = candidate_tree_dot_from_data(candidates, parents, val_scores, pareto_front);

    expect(dot.startsWith('digraph G {')).toBe(true);
    expect(dot.endsWith('}')).toBe(true);
    expect(dot).toContain('    0 [label="');
    expect(dot).toContain('0 -> 1;');
    expect(dot).toContain('0 -> 2;');
    expect(dot).toContain('fillcolor=cyan');
    expect(dot).toContain('tooltip=" "');
    expect(dot).not.toContain('helpful assistant');
  });

  it('generates self-contained HTML with DOT, node metadata, CDN, and tooltip hook', () => {
    const { candidates, parents, val_scores, pareto_front } = sample_data();
    const html = candidate_tree_html_from_data(candidates, parents, val_scores, pareto_front);

    expect(html).toContain('<!DOCTYPE html>');
    expect(html).toContain('</html>');
    expect(html).toContain('digraph G');
    expect(html).toContain('"score"');
    expect(html).toContain('"components"');
    expect(html).toContain('helpful assistant');
    expect(html).toContain('viz-standalone.mjs');
    expect(html).toContain('id="tooltip"');
    expect(html).toContain('showTooltip');
  });

  it('adds GEPAResult convenience methods', () => {
    const { candidates } = sample_data();
    const state = initialize_gepa_state({
      run_dir: null,
      logger: { log: () => undefined },
      seed_candidate: candidates[0]!,
      seed_valset_evaluation: new ValsetEvaluation({
        outputs_by_val_id: new Map([[0, {}], [1, {}]]),
        scores_by_val_id: new Map([[0, 0.5], [1, 0.5]]),
        objective_scores_by_val_id: null,
      }),
      track_best_outputs: false,
      frontier_type: 'instance',
    });
    state.update_state_with_new_program({
      parent_program_idx: [0],
      new_program: candidates[1]!,
      valset_evaluation: new ValsetEvaluation({
        outputs_by_val_id: new Map([[0, {}], [1, {}]]),
        scores_by_val_id: new Map([[0, 0.7], [1, 0.7]]),
        objective_scores_by_val_id: null,
      }),
      run_dir: null,
      num_metric_calls_by_discovery_of_new_program: 2,
    });
    const result = result_from_state(state, {});

    expect(result.candidate_tree_dot()).toContain('0 -> 1;');
    expect(result.candidate_tree_html()).toContain('math tutor');
  });
});
