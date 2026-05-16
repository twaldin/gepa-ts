import type { ChatMessage, LanguageModel } from "./types";

type InputRecord = Record<string, unknown>;
type Prompt = string | ChatMessage[];
type ImageContentPart = { type: "image_url"; image_url: { url: string } };

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

function is_record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function has_openai_content_part_method(value: unknown): value is { to_openai_content_part: () => unknown } {
  return is_record(value) && typeof value["to_openai_content_part"] === "function";
}

function is_image_content_part(value: unknown): value is ImageContentPart {
  return (
    is_record(value) &&
    value["type"] === "image_url" &&
    is_record(value["image_url"]) &&
    typeof value["image_url"]["url"] === "string"
  );
}

function image_content_part(value: unknown): ImageContentPart | null {
  if (has_openai_content_part_method(value)) {
    const part = value.to_openai_content_part();
    return is_image_content_part(part) ? part : null;
  }
  if (is_record(value) && is_image_content_part(value["__gepa_image"])) {
    return value["__gepa_image"];
  }
  if (is_image_content_part(value)) {
    return value;
  }
  return null;
}

function literal_replace_all(input: string, search: string, replacement: string): string {
  return input.split(search).join(replacement);
}

function prompt_renderer(input_dict: PromptInput): Prompt {
  const current_instruction = input_dict.current_instruction_doc;
  if (typeof current_instruction !== "string") {
    throw new TypeError("current_instruction_doc must be a string");
  }

  const dataset = input_dict.dataset_with_feedback;
  if (!Array.isArray(dataset)) {
    throw new TypeError("dataset_with_feedback must be a sequence of records");
  }

  const images: ImageContentPart[] = [];

  function render_value(value: unknown, level = 3): string {
    const image_part = image_content_part(value);
    if (image_part !== null) {
      images.push(image_part);
      return `[IMAGE-${images.length} — see visual content]\n\n`;
    }

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

    if (is_record(value)) {
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

  let formatted_text = dataset.map((sample, i) => convert_sample_to_markdown(sample, i + 1)).join("\n\n");
  if (images.length > 0) {
    formatted_text =
      `The evaluation data below includes visual content (${images.length} image(s)). ` +
      "Analyze both the text and images when suggesting improvements.\n\n" +
      formatted_text;
  }

  const prompt_template = input_dict.prompt_template ?? default_prompt_template;
  validate_prompt_template(prompt_template);

  let prompt = literal_replace_all(prompt_template, "<curr_param>", current_instruction);
  prompt = literal_replace_all(prompt, "<side_info>", formatted_text);
  if (images.length === 0) {
    return prompt;
  }

  return [{ role: "user", content: [{ type: "text", text: prompt }, ...images] }];
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

async function run(lm: LanguageModel, input_dict: PromptInput): Promise<{ new_instruction: string }> {
  const prompt = prompt_renderer(input_dict);
  const lm_out = await lm(prompt);
  return output_extractor(lm_out);
}

async function run_with_metadata(
  lm: LanguageModel,
  input_dict: PromptInput,
): Promise<{ outputs: { new_instruction: string }; prompt: Prompt; lm_output: string }> {
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
