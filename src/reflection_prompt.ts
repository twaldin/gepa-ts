export const optimize_anything_reflection_prompt_template = `I am optimizing a parameter in my system. The current parameter value is:
\`\`\`
<curr_param>
\`\`\`

Below is evaluation data showing how this parameter value performed across multiple test cases. The data contains performance metrics, diagnostic information, and other relevant details from the evaluation:
\`\`\`
<side_info>
\`\`\`

Your task is to propose a new, improved parameter value that can be used as a drop-in replacement for the current one.

Carefully analyze all the evaluation data provided above. Look for patterns that indicate what works and what doesn't. Pay special attention to:
- Performance metrics and how they correlate with parameter behavior
- Recurring issues, errors, or failure patterns across multiple test cases
- Successful patterns or behaviors that should be preserved or enhanced
- Any domain-specific requirements, constraints, or factual information revealed in the evaluation data
- Specific technical details that are crucial for understanding the parameter's role

Based on your analysis, propose a new parameter value that addresses the identified issues while maintaining or improving upon what works well. Your proposal should be directly informed by the patterns and insights from the evaluation data.

Provide the new parameter value within \`\`\` blocks.`;

export function build_reflection_prompt_template({
  objective,
  background,
}: {
  objective?: string | null;
  background?: string | null;
}): string {
  const sections: string[] = [];

  sections.push(
    "You are an expert optimization assistant. Your task is to analyze evaluation feedback and propose an improved version of a system component.",
  );

  if (objective) {
    sections.push(`
## Optimization Goal

${objective}`);
  }

  if (background) {
    sections.push(`
## Domain Context & Constraints

${background}`);
  }

  sections.push(`
## Current Component

The component being optimized:

\`\`\`
<curr_param>
\`\`\`

## Evaluation Results

Performance data from evaluating the current component across test cases:

\`\`\`
<side_info>
\`\`\``);

  const analysis_points: string[] = [];
  if (objective) {
    analysis_points.push(
      "- **Goal alignment**: How well does the current component achieve the stated optimization goal?",
    );
  }
  analysis_points.push(
    "- **Failure patterns**: What specific errors, edge cases, or failure modes appear in the evaluation data?",
    "- **Success patterns**: What behaviors or approaches worked well and should be preserved?",
    "- **Root causes**: What underlying issues explain the observed failures?",
  );
  if (background) {
    analysis_points.push(
      "- **Constraint compliance**: Does the component satisfy all requirements from the domain context?",
    );
  }

  const analysis_section = analysis_points.join("\n");
  const constraint_line = background
    ? "\n4. Adheres to all constraints and requirements from the domain context"
    : "";

  sections.push(`
## Your Task

Analyze the evaluation results systematically:

${analysis_section}

Based on your analysis, propose an improved version that:
1. Addresses the identified failure patterns and root causes
2. Preserves successful behaviors from the current version
3. Makes meaningful improvements rather than superficial changes${constraint_line}`);

  sections.push(`
## Output Format

Provide ONLY the improved version within \`\`\` blocks. The output must be a complete, 
drop-in replacement for the current component (whether it's a prompt, configuration, 
code, or any other parameter type).
Do not include explanations, commentary, or markdown outside the \`\`\` blocks.`);

  return sections.join("\n");
}
