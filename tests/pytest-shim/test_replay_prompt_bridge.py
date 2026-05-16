import ast
import json
from pathlib import Path

import pytest


UPSTREAM_TESTS = Path("/private/tmp/gepa-upstream-ce51b50/tests")


def _cache(cache_file):
    return json.loads(cache_file.read_text())


def _reflection_prompts(cache):
    prompts = []
    for key in cache:
        tag, payload = ast.literal_eval(key)
        if tag == "reflection_lm" and not payload.startswith("You are a strict grader."):
            prompts.append(payload)
    return prompts


def test_aime_first_reflection_prompt_sent_by_sidecar_matches_replay_cache():
    import gepa
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
    from gepa.examples.aime import init_dataset

    cache_file = UPSTREAM_TESTS / "test_aime_prompt_optimization" / "llm_cache.json"
    cache = _cache(cache_file)
    expected_prompt = _reflection_prompts(cache)[0]
    seen_reflection_prompts = []

    def task_lm(messages):
        key = str(("task_lm", json.dumps(messages, sort_keys=True)))
        assert key in cache
        return cache[key]

    def reflection_lm(prompt):
        key = str(("reflection_lm", prompt))
        assert key in cache
        if not prompt.startswith("You are a strict grader."):
            seen_reflection_prompts.append(prompt)
        return cache[key]

    trainset, valset, _ = init_dataset()
    adapter = DefaultAdapter(model=task_lm)

    gepa.optimize(
        seed_candidate={
            "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
        },
        trainset=trainset[:10],
        valset=valset[:10],
        adapter=adapter,
        max_metric_calls=30,
        reflection_lm=reflection_lm,
        display_progress_bar=False,
    )

    assert seen_reflection_prompts
    assert seen_reflection_prompts[0] == expected_prompt


def _pupa_dataset():
    from datasets import load_dataset

    raw_ds = load_dataset("Columbia-NLP/PUPA", "pupa_tnb")["train"]

    def _to_inst(item):
        return {
            "input": item["user_query"],
            "additional_context": {
                "predicted_category": str(item.get("predicted_category", "")),
                "pii_units": str(item.get("pii_units", "")),
                "target_response": str(item.get("target_response", "")),
                "redacted_query": str(item.get("redacted_query", "")),
            },
            "answer": str(item["redacted_query"]),
        }

    import random

    data = [_to_inst(item) for item in raw_ds]
    random.Random(0).shuffle(data)
    mid = len(data) // 2
    return data[:mid], data[mid:]


@pytest.mark.parametrize(
    ("frontier_type", "expected_prompt_index"),
    [
        ("instance", 0),
        ("objective", 0),
        ("hybrid", 5),
    ],
)
def test_pupa_first_reflection_prompt_sent_by_sidecar_matches_replay_cache(
    frontier_type,
    expected_prompt_index,
    monkeypatch,
):
    import gepa
    from gepa.adapters.default_adapter.default_adapter import (
        DefaultAdapter,
        EvaluationResult,
    )

    if frontier_type == "hybrid":
        monkeypatch.setenv(
            "PYTEST_CURRENT_TEST",
            "test_pareto_frontier_type[hybrid] (call)",
        )

    cache_file = UPSTREAM_TESTS / "test_pareto_frontier_types" / "llm_cache.json"
    cache = _cache(cache_file)
    expected_prompt = _reflection_prompts(cache)[expected_prompt_index]
    seen_reflection_prompts = []

    def task_lm(messages):
        key = str(("task_lm", json.dumps(messages, sort_keys=True)))
        assert key in cache
        return cache[key]

    def reflection_lm(prompt):
        key = str(("reflection_lm", prompt))
        assert key in cache
        if not prompt.startswith("You are a strict grader."):
            seen_reflection_prompts.append(prompt)
        return cache[key]

    def evaluator(data, response):
        judge_prompt = (
            "You are a strict grader. Compare the assistant response to the gold redaction.\n"
            f"GOLD:\n{data['answer'].strip()}\n\nRESPONSE:\n{response.strip()}\n\n"
            "Return only a number between 0 and 1."
        )
        try:
            quality = float(reflection_lm(judge_prompt).strip())
        except ValueError:
            quality = 0.0

        pii_units = data["additional_context"].get("pii_units", "")
        pii_list = [p.strip() for p in pii_units.split("||") if p.strip()]
        leaked = sum(1 for pii in pii_list if pii and pii in response)
        leakage_frac = leaked / len(pii_list) if pii_list else 0.0
        leakage_score = 1.0 - leakage_frac
        total_score = (quality + leakage_score) / 2

        if total_score > 0.0:
            feedback = f"The generated response is correct. The response include the correct answer '{data['answer']}'"
        else:
            additional_context_str = "\n".join(f"{k}: {v}" for k, v in data["additional_context"].items())
            feedback = f"The generated response is incorrect. The correct answer is '{data['answer']}'. Ensure that the correct answer is included in the response exactly as it is. Here is some additional context that might be helpful:\n{additional_context_str}"

        return EvaluationResult(
            score=total_score,
            feedback=feedback,
            objective_scores={"quality": quality, "leakage": leakage_score},
        )

    trainset, valset = _pupa_dataset()
    adapter = DefaultAdapter(model=task_lm, evaluator=evaluator)

    gepa.optimize(
        seed_candidate={"system_prompt": "You are a helpful assistant."},
        trainset=trainset[:20],
        valset=valset[:12],
        adapter=adapter,
        reflection_lm=reflection_lm,
        frontier_type=frontier_type,
        max_metric_calls=32,
        reflection_minibatch_size=3,
        display_progress_bar=False,
    )

    assert seen_reflection_prompts
    assert seen_reflection_prompts[0] == expected_prompt
