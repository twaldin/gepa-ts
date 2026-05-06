type InputRecord = Record<string, unknown>;

type PromptInput = {
  current_instruction_doc: string;
  dataset_with_feedback: ReadonlyArray<InputRecord>;
  prompt_template: string | null;
};

const default_prompt_template = `I provided an assistant with the following instructions to perform a task for me:
\`\`\`
<curr_param>
\`\`\`

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
\`\`\`
<side_info>
\`\`\`

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within \`\`\` blocks.`;

function validate_prompt_template(prompt_template: string | null): void {
  if (prompt_template === null) {
    return;
  }
  const missing_placeholders = ["<curr_param>", "<side_info>"].filter(
    (placeholder) => !prompt_template.includes(placeholder),
  );
  if (missing_placeholders.length > 0) {
    throw new Error(`Missing placeholder(s) in prompt template: ${missing_placeholders.join(", ")}`);
  }
}

function is_image_like(value: unknown): boolean {
  return typeof value === "object" && value !== null && "to_openai_content_part" in value;
}

function contains_image_like(value: unknown): boolean {
  if (is_image_like(value)) {
    return true;
  }
  if (Array.isArray(value)) {
    return value.some((item) => contains_image_like(item));
  }
  if (typeof value === "object" && value !== null) {
    return Object.values(value).some((item) => contains_image_like(item));
  }
  return false;
}

function prompt_renderer(input_dict: PromptInput): string {
  const current_instruction = input_dict.current_instruction_doc;
  if (typeof current_instruction !== "string") {
    throw new TypeError("current_instruction_doc must be a string");
  }

  const dataset = input_dict.dataset_with_feedback;
  if (!Array.isArray(dataset)) {
    throw new TypeError("dataset_with_feedback must be a sequence of records");
  }

  if (contains_image_like(dataset)) {
    throw new Error("Image content is not supported in v1 prompt_renderer");
  }

  function render_value(value: unknown, level = 3): string {
    if (Array.isArray(value)) {
      let out = "";
      value.forEach((item, i) => {
        out += `${"#".repeat(level)} Item ${i + 1}\n`;
        out += render_value(item, Math.min(level + 1, 6));
      });
      if (value.length === 0) {
        out += "\n";
      }
      return out;
    }

    if (typeof value === "object" && value !== null) {
      let out = "";
      Object.entries(value).forEach(([key, nested_value]) => {
        out += `${"#".repeat(level)} ${key}\n`;
        out += render_value(nested_value, Math.min(level + 1, 6));
      });
      if (Object.keys(value).length === 0) {
        out += "\n";
      }
      return out;
    }

    return `${String(value).trim()}\n\n`;
  }

  function convert_sample_to_markdown(sample: InputRecord, examplenum: number): string {
    let out = `# Example ${examplenum}\n`;
    Object.entries(sample).forEach(([key, val]) => {
      out += `## ${key}\n`;
      out += render_value(val, 3);
    });
    return out;
  }

  const formatted_text = dataset.map((sample, i) => convert_sample_to_markdown(sample, i + 1)).join("\n\n");

  const prompt_template = input_dict.prompt_template ?? default_prompt_template;
  validate_prompt_template(prompt_template);

  let prompt = prompt_template.replace("<curr_param>", current_instruction);
  prompt = prompt.replace("<side_info>", formatted_text);
  return prompt;
}

function output_extractor(lm_out: string): { new_instruction: string } {
  const start = lm_out.indexOf("```") + 3;
  const end = lm_out.lastIndexOf("```");

  if (start >= end) {
    const stripped = lm_out.trim();
    if (stripped.startsWith("```")) {
      const match = lm_out.match(/^```\S*\n?/);
      if (match) {
        return { new_instruction: lm_out.slice(match[0].length).trim() };
      }
    } else if (stripped.endsWith("```")) {
      return { new_instruction: stripped.slice(0, -3).trim() };
    }
    return { new_instruction: stripped };
  }

  let content = lm_out.slice(start, end);
  const match = content.match(/^\S*\n/);
  if (match) {
    content = content.slice(match[0].length);
  }

  return { new_instruction: content.trim() };
}

async function run(lm: (prompt: string) => Promise<string>, input_dict: PromptInput): Promise<{ new_instruction: string }> {
  const prompt = prompt_renderer(input_dict);
  const lm_out = await lm(prompt);
  return output_extractor(lm_out);
}

async function run_with_metadata(
  lm: (prompt: string) => Promise<string>,
  input_dict: PromptInput,
): Promise<{ outputs: { new_instruction: string }; prompt: string; lm_output: string }> {
  const prompt = prompt_renderer(input_dict);
  const lm_output = await lm(prompt);
  return {
    outputs: output_extractor(lm_output),
    prompt,
    lm_output,
  };
}

export const InstructionProposalSignature = {
  default_prompt_template,
  validate_prompt_template,
  prompt_renderer,
  output_extractor,
  run,
  run_with_metadata,
};
