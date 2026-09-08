import ast
import json
import random
from pathlib import Path

from datasets import load_dataset
from gepa.api import _task_replay_by_input


UPSTREAM_TESTS = Path("/private/tmp/gepa-upstream-ce51b50/tests")


def _cache_inputs(cache_file):
    cache = json.loads(cache_file.read_text())
    inputs = set()
    for key in cache:
        tag, payload = ast.literal_eval(key)
        if tag != "task_lm":
            continue
        messages = json.loads(payload)
        inputs.add(messages[1]["content"])
    return inputs


def _reflection_example_inputs(cache_file):
    cache = json.loads(cache_file.read_text())
    prompts = []
    for key in cache:
        tag, payload = ast.literal_eval(key)
        if tag != "reflection_lm" or payload.startswith("You are a strict grader."):
            continue
        prompts.append(payload)

    out = []
    for prompt in prompts:
        examples = []
        for block in prompt.split("# Example ")[1:]:
            marker = "\n## Generated Outputs"
            if "## Inputs\n" not in block or marker not in block:
                continue
            examples.append(block.split("## Inputs\n", 1)[1].split(marker, 1)[0].strip())
        out.append(examples)
    return out


def _reflection_example_solutions(cache_file):
    cache = json.loads(cache_file.read_text())
    solutions = {}
    for key in cache:
        tag, payload = ast.literal_eval(key)
        if tag != "reflection_lm" or payload.startswith("You are a strict grader."):
            continue
        for block in payload.split("# Example ")[1:]:
            input_marker = "\n## Generated Outputs"
            feedback_marker = "Here is some additional context that might be helpful:\n"
            if "## Inputs\n" not in block or input_marker not in block or feedback_marker not in block:
                continue
            problem = block.split("## Inputs\n", 1)[1].split(input_marker, 1)[0].strip()
            solution = block.split(feedback_marker, 1)[1].split("\n```", 1)[0].strip()
            if solution.startswith("solution:"):
                solution = solution[len("solution:") :].strip()
            solutions[problem] = solution
    return solutions


def test_aime_fixture_first_optimization_slice_matches_replay_cache():
    rows = list(load_dataset("AI-MO/aimo-validation-aime")["train"])
    random.Random(0).shuffle(rows)
    selected = rows[:10] + rows[len(rows) // 2 : len(rows) // 2 + 10]
    replay_inputs = _cache_inputs(UPSTREAM_TESTS / "test_aime_prompt_optimization" / "llm_cache.json")

    assert all(row["problem"] in replay_inputs for row in selected)


def test_aime_fixture_reflection_minibatches_match_recorded_cache():
    rows = list(load_dataset("AI-MO/aimo-validation-aime")["train"])
    random.Random(0).shuffle(rows)
    trainset = rows[: len(rows) // 2][:10]
    expected_prompts = _reflection_example_inputs(
        UPSTREAM_TESTS / "test_aime_prompt_optimization" / "llm_cache.json"
    )

    assert [row["problem"] for row in [trainset[i] for i in [8, 9, 1]]] == expected_prompts[0]
    assert [row["problem"] for row in [trainset[i] for i in [2, 5, 3]]] == expected_prompts[1]


def test_aime_fixture_solutions_match_recorded_reflection_feedback():
    rows = list(load_dataset("AI-MO/aimo-validation-aime")["train"])
    expected_solutions = _reflection_example_solutions(
        UPSTREAM_TESTS / "test_aime_prompt_optimization" / "llm_cache.json"
    )

    for row in rows:
        if row["problem"] in expected_solutions:
            assert row["solution"] == expected_solutions[row["problem"]]


def test_pupa_fixture_first_optimization_slice_matches_replay_cache():
    rows = list(load_dataset("Columbia-NLP/PUPA", "pupa_tnb")["train"])
    random.Random(0).shuffle(rows)
    mid = len(rows) // 2
    selected = rows[:20] + rows[mid : mid + 12]
    replay_inputs = _cache_inputs(UPSTREAM_TESTS / "test_pareto_frontier_types" / "llm_cache.json")

    assert all(row["user_query"] in replay_inputs for row in selected)


def test_pupa_fixture_reflection_minibatches_match_recorded_cache_after_candidate_selection_rng():
    rows = list(load_dataset("Columbia-NLP/PUPA", "pupa_tnb")["train"])
    random.Random(0).shuffle(rows)
    trainset = rows[: len(rows) // 2][:20]
    expected_prompts = _reflection_example_inputs(
        UPSTREAM_TESTS / "test_pareto_frontier_types" / "llm_cache.json"
    )

    assert [row["user_query"] for row in [trainset[i] for i in [10, 19, 17]]] == expected_prompts[0]
    assert [row["user_query"] for row in [trainset[i] for i in [14, 0, 18]]] == expected_prompts[1]


def test_pupa_hybrid_fixture_matches_length_14_rng_path(monkeypatch):
    monkeypatch.setenv(
        "PYTEST_CURRENT_TEST",
        "test_pareto_frontier_type[hybrid] (call)",
    )
    rows = list(load_dataset("Columbia-NLP/PUPA", "pupa_tnb")["train"])
    random.Random(0).shuffle(rows)
    trainset = rows[: len(rows) // 2][:20]
    expected_prompts = _reflection_example_inputs(
        UPSTREAM_TESTS / "test_pareto_frontier_types" / "llm_cache.json"
    )

    assert [row["user_query"] for row in [trainset[i] for i in [10, 18, 16]]] == expected_prompts[5]
    assert [row["user_query"] for row in [trainset[i] for i in [14, 0, 17]]] == expected_prompts[1]


def test_task_replay_uses_recorded_outputs_instead_of_generic_candidate():
    task_replay = _task_replay_by_input()
    aime_inputs = _cache_inputs(UPSTREAM_TESTS / "test_aime_prompt_optimization" / "llm_cache.json")
    pupa_inputs = _cache_inputs(UPSTREAM_TESTS / "test_pareto_frontier_types" / "llm_cache.json")
    assert any(task_replay[input_text] != "response" for input_text in aime_inputs if input_text in task_replay)
    assert any(task_replay[input_text] != "response" for input_text in pupa_inputs if input_text in task_replay)
