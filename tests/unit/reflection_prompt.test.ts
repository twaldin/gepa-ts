import { describe, expect, test } from "vitest";
import {
  build_reflection_prompt_template,
  optimize_anything_reflection_prompt_template,
} from "../../src/reflection_prompt";

describe("optimize_anything_reflection_prompt_template", () => {
  test("contains required placeholders", () => {
    expect(optimize_anything_reflection_prompt_template.length).toBeGreaterThan(0);
    expect(optimize_anything_reflection_prompt_template).toContain("<curr_param>");
    expect(optimize_anything_reflection_prompt_template).toContain("<side_info>");
  });
});

describe("build_reflection_prompt_template", () => {
  test("objective section included", () => {
    const prompt = build_reflection_prompt_template({ objective: "Optimize X" });
    expect(prompt).toContain("## Optimization Goal");
    expect(prompt).toContain("Optimize X");
  });

  test("background section included", () => {
    const prompt = build_reflection_prompt_template({ background: "Use Y" });
    expect(prompt).toContain("## Domain Context & Constraints");
    expect(prompt).toContain("Use Y");
  });

  test("works with both omitted", () => {
    const prompt = build_reflection_prompt_template({});
    expect(prompt).toContain("<curr_param>");
    expect(prompt).toContain("<side_info>");
  });

  test("section order: goal then background then current component", () => {
    const prompt = build_reflection_prompt_template({ objective: "Optimize X", background: "Use Y" });
    const goal_index = prompt.indexOf("## Optimization Goal");
    const background_index = prompt.indexOf("## Domain Context & Constraints");
    const current_index = prompt.indexOf("## Current Component");

    expect(goal_index).toBeGreaterThan(-1);
    expect(background_index).toBeGreaterThan(-1);
    expect(current_index).toBeGreaterThan(-1);
    expect(goal_index).toBeLessThan(background_index);
    expect(background_index).toBeLessThan(current_index);
  });
});
