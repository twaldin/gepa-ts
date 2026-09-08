import { describe, expect, test } from "vitest";
import { InstructionProposalSignature } from "../../src/instruction_proposal";

describe("InstructionProposalSignature.validate_prompt_template", () => {
  test("null does not throw", () => {
    expect(() => InstructionProposalSignature.validate_prompt_template(null)).not.toThrow();
  });

  test("missing <side_info> throws", () => {
    expect(() => InstructionProposalSignature.validate_prompt_template("only <curr_param>")).toThrow(
      "<side_info>",
    );
  });

  test("missing <curr_param> throws", () => {
    expect(() => InstructionProposalSignature.validate_prompt_template("only <side_info>")).toThrow(
      "<curr_param>",
    );
  });

  test("both placeholders does not throw", () => {
    expect(() =>
      InstructionProposalSignature.validate_prompt_template("has both <curr_param> and <side_info>"),
    ).not.toThrow();
  });
});

describe("InstructionProposalSignature.prompt_renderer", () => {
  test("renders simple dataset and current instruction", () => {
    const prompt = InstructionProposalSignature.prompt_renderer({
      current_instruction_doc: "do X",
      dataset_with_feedback: [{ Inputs: "a", Outputs: "b", Feedback: "c" }],
      prompt_template: null,
    });

    expect(prompt).toContain("# Example 1");
    expect(prompt).toContain("## Inputs");
    expect(prompt).toContain("## Outputs");
    expect(prompt).toContain("## Feedback");
    expect(prompt).toContain("do X");
    expect(prompt).toContain("a");
    expect(prompt).toContain("b");
    expect(prompt).toContain("c");
  });

  test("matches upstream default prompt rendering exactly for text-only samples", () => {
    const prompt = InstructionProposalSignature.prompt_renderer({
      current_instruction_doc: "do X",
      dataset_with_feedback: [{ Inputs: "a", Outputs: "b", Feedback: "c" }],
      prompt_template: null,
    });

    expect(prompt).toBe(
      "I provided an assistant with the following instructions to perform a task for me:\n" +
        "```\n" +
        "do X\n" +
        "```\n" +
        "\n" +
        "The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:\n" +
        "```\n" +
        "# Example 1\n" +
        "## Inputs\n" +
        "a\n" +
        "\n" +
        "## Outputs\n" +
        "b\n" +
        "\n" +
        "## Feedback\n" +
        "c\n" +
        "\n" +
        "\n" +
        "```\n" +
        "\n" +
        "Your task is to write a new instruction for the assistant.\n" +
        "\n" +
        "Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.\n" +
        "\n" +
        "Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.\n" +
        "\n" +
        "Provide the new instructions within ``` blocks.",
    );
  });

  test("nested dict/list headers deepen and cap at level 6", () => {
    const prompt = InstructionProposalSignature.prompt_renderer({
      current_instruction_doc: "do X",
      dataset_with_feedback: [
        {
          Inputs: {
            l1: {
              l2: {
                l3: {
                  l4: {
                    l5: {
                      l6: "value",
                    },
                  },
                },
              },
            },
          },
        },
      ],
      prompt_template: "<curr_param>\n<side_info>",
    });

    expect(prompt).toContain("### l1");
    expect(prompt).toContain("#### l2");
    expect(prompt).toContain("##### l3");
    expect(prompt).toContain("###### l4");
    expect(prompt).toContain("###### l5");
    expect(prompt).toContain("###### l6");
  });

  test("empty list and dict add extra newline", () => {
    const prompt = InstructionProposalSignature.prompt_renderer({
      current_instruction_doc: "do X",
      dataset_with_feedback: [{ Inputs: [], Outputs: {}, Feedback: "ok" }],
      prompt_template: "<curr_param>\n<side_info>",
    });

    expect(prompt).toContain("## Inputs\n\n");
    expect(prompt).toContain("## Outputs\n\n");
  });

  test("keeps Scores (Higher is Better) key unchanged", () => {
    const prompt = InstructionProposalSignature.prompt_renderer({
      current_instruction_doc: "do X",
      dataset_with_feedback: [{ "Scores (Higher is Better)": { acc: 0.5 } }],
      prompt_template: "<curr_param>\n<side_info>",
    });

    expect(prompt).toContain("## Scores (Higher is Better)");
  });

  test("replaces all placeholder occurrences", () => {
    const prompt = InstructionProposalSignature.prompt_renderer({
      current_instruction_doc: "do X",
      dataset_with_feedback: [{ Inputs: "a" }],
      prompt_template: "<curr_param> :: <curr_param> :: <side_info> :: <side_info>",
    });

    expect(prompt).toContain("do X :: do X");
    expect(prompt).not.toContain("<curr_param>");
    expect(prompt).not.toContain("<side_info>");
  });

  test("inserts replacement text literally when feedback contains JavaScript replacement tokens", () => {
    const prompt = InstructionProposalSignature.prompt_renderer({
      current_instruction_doc: "Solve math problems ending with $& and $` tokens.",
      dataset_with_feedback: [
        {
          Feedback: "Of course, $99$ can't be made of just $8$'s.",
        },
      ],
      prompt_template: "A:<curr_param>\nB:<side_info>\nEND",
    });

    expect(prompt).toContain("Solve math problems ending with $& and $` tokens.");
    expect(prompt).toContain("Of course, $99$ can't be made of just $8$'s.");
    expect(prompt).not.toContain("``` blocks.s");
    expect(prompt).not.toContain("B:\n## Feedback\nOf course, $99$ can't be made of just $8\nENDs.");
  });

  test("matches upstream custom prompt rendering exactly when feedback contains replacement tokens", () => {
    const prompt = InstructionProposalSignature.prompt_renderer({
      current_instruction_doc: "Solve math problems ending with $& and $` tokens.",
      dataset_with_feedback: [
        {
          Feedback: "Of course, $99$ can't be made of just $8$'s.",
        },
      ],
      prompt_template: "A:<curr_param>\nB:<side_info>\nEND",
    });

    expect(prompt).toBe(
      "A:Solve math problems ending with $& and $` tokens.\n" +
        "B:# Example 1\n" +
        "## Feedback\n" +
        "Of course, $99$ can't be made of just $8$'s.\n" +
        "\n" +
        "\n" +
        "END",
    );
  });

  test("renders OpenAI multimodal messages when records include Python shim image sentinels", () => {
    const prompt = InstructionProposalSignature.prompt_renderer({
      current_instruction_doc: "inspect image",
      dataset_with_feedback: [
        {
          Input: "draw a red circle",
          Rendering: {
            __gepa_image: {
              type: "image_url",
              image_url: { url: "data:image/png;base64,abc123" },
            },
          },
          Feedback: "Circle is blue instead of red",
        },
      ],
      prompt_template: null,
    });

    expect(Array.isArray(prompt)).toBe(true);
    if (!Array.isArray(prompt)) {
      throw new Error("expected multimodal prompt");
    }
    expect(prompt[0]?.role).toBe("user");
    const content = prompt[0]?.content;
    expect(Array.isArray(content)).toBe(true);
    if (!Array.isArray(content)) {
      throw new Error("expected multimodal content");
    }
    expect(content[0]).toMatchObject({ type: "text" });
    expect(String((content[0] as { text: string }).text)).toContain("[IMAGE-1 — see visual content]");
    expect(String((content[0] as { text: string }).text)).toContain("Circle is blue");
    expect(content[1]).toEqual({
      type: "image_url",
      image_url: { url: "data:image/png;base64,abc123" },
    });
  });

  test("run_with_metadata passes multimodal prompt directly to the reflection lm", async () => {
    const seen: unknown[] = [];
    const result = await InstructionProposalSignature.run_with_metadata(
      async (prompt) => {
        seen.push(prompt);
        return "```\nimproved\n```";
      },
      {
        current_instruction_doc: "inspect image",
        dataset_with_feedback: [
          {
            Frames: [
              {
                __gepa_image: {
                  type: "image_url",
                  image_url: { url: "https://example.test/frame.png" },
                },
              },
            ],
          },
        ],
        prompt_template: "<curr_param>\n<side_info>",
      },
    );

    expect(result.outputs).toEqual({ new_instruction: "improved" });
    expect(seen).toHaveLength(1);
    expect(Array.isArray(seen[0])).toBe(true);
    expect(result.prompt).toBe(seen[0]);
  });
});

describe("InstructionProposalSignature.output_extractor", () => {
  test("critical fenced candidate case", () => {
    expect(InstructionProposalSignature.output_extractor("```\ncandidate\n```")).toEqual({
      new_instruction: "candidate",
    });
  });

  test("strips language tag", () => {
    expect(InstructionProposalSignature.output_extractor("```python\ndef f(): pass\n```")).toEqual({
      new_instruction: "def f(): pass",
    });
  });

  test("multiline content", () => {
    expect(InstructionProposalSignature.output_extractor("```\nmulti\nline\n```")).toEqual({
      new_instruction: "multi\nline",
    });
  });

  test("only opening fence", () => {
    expect(InstructionProposalSignature.output_extractor("```\ntext")).toEqual({
      new_instruction: "text",
    });
  });

  test("only closing fence", () => {
    expect(InstructionProposalSignature.output_extractor("text\n```")).toEqual({
      new_instruction: "text",
    });
  });

  test("no fences", () => {
    expect(InstructionProposalSignature.output_extractor("no fences at all")).toEqual({
      new_instruction: "no fences at all",
    });
  });

  test("just three backticks", () => {
    expect(InstructionProposalSignature.output_extractor("```")).toEqual({
      new_instruction: "",
    });
  });
});
