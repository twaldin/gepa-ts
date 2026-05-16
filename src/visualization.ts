import { find_dominator_programs } from './utils.js';
import type { Candidate, DataId, ProgramIdx } from './types.js';

function escape_html(text: string): string {
  return text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#x27;');
}

function pareto_record(pareto_front_programs: Map<DataId, Set<ProgramIdx>> | Record<string | number, Set<ProgramIdx>>): Record<string | number, Set<ProgramIdx>> {
  if (pareto_front_programs instanceof Map) {
    return Object.fromEntries([...pareto_front_programs.entries()].map(([key, value]) => [key, new Set(value)]));
  }
  return pareto_front_programs;
}

export function candidate_tree_dot_from_data(
  candidates: Candidate[],
  parents: Array<Array<ProgramIdx | null>>,
  val_scores: number[],
  pareto_front_programs: Map<DataId, Set<ProgramIdx>> | Record<string | number, Set<ProgramIdx>>,
): string {
  const n = candidates.length;
  const best_idx = n > 0
    ? val_scores.reduce((best_i, score, i) => score > (val_scores[best_i] ?? Number.NEGATIVE_INFINITY) ? i : best_i, 0)
    : 0;
  const pareto_record_value = pareto_record(pareto_front_programs);
  const dominator_ids = new Set(find_dominator_programs(pareto_record_value, val_scores));

  const dot_lines = [
    'digraph G {',
    '    rankdir=TB;',
    '    node [style=filled, shape=circle, fontsize=14, width=0.6, height=0.6];',
  ];

  for (let idx = 0; idx < n; idx += 1) {
    const score = val_scores[idx] ?? 0;
    const label = `${idx}\\n(${score.toFixed(2)})`;
    const color = idx === best_idx ? 'cyan' : dominator_ids.has(idx) ? 'orange' : 'lightgray';
    dot_lines.push(`    ${idx} [label="${label}", fillcolor=${color}, tooltip=" "];`);
  }

  for (let child = 0; child < parents.length; child += 1) {
    for (const parent of parents[child] ?? []) {
      if (parent !== null) {
        dot_lines.push(`    ${parent} -> ${child};`);
      }
    }
  }

  dot_lines.push('}');
  return dot_lines.join('\n');
}

export function candidate_tree_html_from_data(
  candidates: Candidate[],
  parents: Array<Array<ProgramIdx | null>>,
  val_scores: number[],
  pareto_front_programs: Map<DataId, Set<ProgramIdx>> | Record<string | number, Set<ProgramIdx>>,
): string {
  const pareto_record_value = pareto_record(pareto_front_programs);
  const dominator_ids = new Set(find_dominator_programs(pareto_record_value, val_scores));
  const best_idx = candidates.length > 0
    ? val_scores.reduce((best_i, score, i) => score > (val_scores[best_i] ?? Number.NEGATIVE_INFINITY) ? i : best_i, 0)
    : 0;
  const nodes = candidates.map((candidate, idx) => {
    const parent_str = (parents[idx] ?? []).filter((parent) => parent !== null).join(', ') || 'seed';
    const role = idx === best_idx ? 'Best' : dominator_ids.has(idx) ? 'Pareto Front' : idx === 0 ? 'Seed' : '';
    return {
      idx,
      score: Number((val_scores[idx] ?? 0).toFixed(4)),
      parents: parent_str,
      role,
      components: Object.fromEntries(Object.entries(candidate).sort(([a], [b]) => a.localeCompare(b))),
    };
  });

  const dot_string = candidate_tree_dot_from_data(candidates, parents, val_scores, pareto_record_value);
  const nodes_json = JSON.stringify(nodes);
  const dot_json = JSON.stringify(dot_string);
  const escaped_nodes = escape_html(nodes_json);

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GEPA Candidate Tree</title>
<script type="module" src="https://cdn.jsdelivr.net/npm/@viz-js/viz@3.12.0/lib/viz-standalone.mjs"></script>
</head>
<body>
<div id="graph-container"></div>
<div id="tooltip"></div>
<script>
const dot = ${dot_json};
const nodes = ${nodes_json};
document.getElementById("graph-container").textContent = dot;
function showTooltip(idx) {
  const node = nodes[idx];
  const tooltip = document.getElementById("tooltip");
  tooltip.textContent = JSON.stringify(node, null, 2);
}
window.showTooltip = showTooltip;
</script>
<script type="application/json" id="node-data">${escaped_nodes}</script>
</body>
</html>`;
}
