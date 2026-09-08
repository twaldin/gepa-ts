export type DSPyProgramProposalInput = {
  curr_program: string;
  dataset_with_feedback: Array<Record<string, unknown>>;
};

export class DSPyProgramProposalSignature {
  static readonly input_keys = ['curr_program', 'dataset_with_feedback'];
  static readonly output_keys = ['new_program'];

  static readonly prompt_template = `I am trying to solve a task using the DSPy framework. Here's a comprehensive overview of DSPy concepts to guide your improvements:

Signatures define tasks declaratively through input/output fields and instructions.
Modules specify how to solve the task and compose LM calls or Python logic.

Here's my current code:
\`\`\`
<curr_program>
\`\`\`

Here is the execution trace of the current code on example inputs, their outputs, and detailed feedback on improvements:
\`\`\`
<dataset_with_feedback>
\`\`\`

Assignment:
- Think step-by-step about failure modes, strengths, and opportunities.
- Create a concise checklist outlining your improvement plan.
- Then propose a drop-in replacement Python script that assigns an improved \`program\` object.
- Output everything in a single code block using triple backticks.`;

  static prompt_renderer(input_dict: DSPyProgramProposalInput): string {
    const curr_program = input_dict.curr_program;
    if (typeof curr_program !== 'string') {
      throw new TypeError('curr_program must be a string');
    }
    const dataset = input_dict.dataset_with_feedback;
    if (!Array.isArray(dataset)) {
      throw new TypeError('dataset_with_feedback must be a list');
    }
    return DSPyProgramProposalSignature.prompt_template
      .replace('<curr_program>', curr_program)
      .replace('<dataset_with_feedback>', format_samples(dataset));
  }

  static output_extractor(lm_out: string): { new_program: string } {
    let new_program: string;
    if (lm_out.split('```').length - 1 >= 2) {
      const start = lm_out.indexOf('```');
      const end = lm_out.lastIndexOf('```');
      if (start >= end || start === -1 || end === -1) {
        new_program = lm_out;
      } else {
        new_program = lm_out.slice(start + 3, end).trim();
      }
    } else {
      new_program = lm_out.trim();
      if (new_program.startsWith('```')) {
        new_program = new_program.slice(3);
      }
      if (new_program.endsWith('```')) {
        new_program = new_program.slice(0, -3);
      }
    }
    return { new_program };
  }

  static async run({
    lm,
    input_dict,
  }: {
    lm: (prompt: string) => string | Promise<string>;
    input_dict: DSPyProgramProposalInput;
  }): Promise<{ new_program: string }> {
    const prompt = DSPyProgramProposalSignature.prompt_renderer(input_dict);
    const output = await lm(prompt);
    return DSPyProgramProposalSignature.output_extractor(output);
  }
}

function format_samples(samples: Array<Record<string, unknown>>): string {
  return samples.map((sample) => format_value(sample, 0).trimEnd()).join('\n');
}

function format_value(value: unknown, indent: number): string {
  const prefix = ' '.repeat(indent);
  if (Array.isArray(value)) {
    return value.map((item) => `${prefix}- ${format_value(item, indent + 2).trimStart()}`).join('\n') + '\n';
  }
  if (typeof value === 'object' && value !== null) {
    return Object.entries(value)
      .map(([key, child]) => {
        if (typeof child === 'object' && child !== null) {
          return `${prefix}${key}:\n${format_value(child, indent + 2)}`;
        }
        return `${prefix}${key}: ${String(child)}\n`;
      })
      .join('');
  }
  return `${prefix}${String(value)}\n`;
}
