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
