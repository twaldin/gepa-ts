import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any


class OptimizeAnythingAdapter:
    def __init__(
        self,
        evaluator,
        parallel: bool = False,
        refiner_config=None,
        cache_mode: str = "off",
        **kwargs,
    ) -> None:
        self.evaluator = evaluator
        self.refiner_config = refiner_config
        self.cache_mode = cache_mode

    def _call_evaluator(self, candidate: dict[str, str], example: Any):
        safe_example = None if type(example) is object else example
        result = self.evaluator(candidate, example=safe_example)
        if isinstance(result, tuple) and len(result) == 3:
            score, output, side_info = result
        elif isinstance(result, tuple) and len(result) == 2:
            score, side_info = result
            output = None
        else:
            score = result
            output = None
            side_info = {}
        return score, output, dict(side_info or {})

    def _parse_refinement(self, text: str) -> dict[str, str]:
        text = str(text).strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            first_newline = text.find("\n")
            if first_newline >= 0:
                text = text[first_newline + 1 :]
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("refiner output must be a JSON dict")
        return {str(k): str(v) for k, v in parsed.items() if k != "refiner_prompt"}

    def _evaluate_single_with_refinement(self, candidate: dict[str, str], example: Any):
        base_candidate = deepcopy(candidate)
        score, output, side_info = self._call_evaluator(base_candidate, example)
        attempts = [
            {
                "iteration": 0,
                "candidate": deepcopy(base_candidate),
                "score": score,
                "side_info": deepcopy(side_info),
            }
        ]

        best_score = score
        best_output = output
        best_side_info = dict(side_info)
        best_scores = dict(side_info.get("scores", {}) or {})

        refiner_lm = getattr(self.refiner_config, "refiner_lm", None)
        max_refinements = getattr(self.refiner_config, "max_refinements", 1) or 1

        for iteration in range(1, max_refinements + 1):
            if refiner_lm is None:
                break
            try:
                prompt = json.dumps(
                    {
                        "candidate": base_candidate,
                        "side_info": best_side_info,
                        "instruction": base_candidate.get("refiner_prompt", ""),
                    }
                )
                refined = self._parse_refinement(refiner_lm(prompt))
                refined_candidate = {**base_candidate, **refined}
                refined_score, refined_output, refined_side_info = self._call_evaluator(refined_candidate, example)
                attempts.append(
                    {
                        "iteration": iteration,
                        "candidate": deepcopy(refined_candidate),
                        "score": refined_score,
                        "side_info": deepcopy(refined_side_info),
                    }
                )
                if refined_score >= best_score:
                    best_score = refined_score
                    best_output = refined_output
                    best_side_info = dict(refined_side_info)
                    best_scores = dict(refined_side_info.get("scores", {}) or best_scores)
                    base_candidate = refined_candidate
            except Exception as exc:
                attempts.append({"iteration": iteration, "error": str(exc)})

        merged_side_info = dict(best_side_info)
        merged_side_info["refiner_prompt_specific_info"] = {
            "scores": best_scores,
            "Attempts": attempts,
        }
        return best_score, best_output, merged_side_info

    def evaluate(self, batch, candidate: dict[str, str], capture_traces: bool = True, opt_states=None):
        outputs = []
        scores = []
        trajectories = []
        objective_scores = []
        side_infos = []
        for example in batch:
            score, output, side_info = self._evaluate_single_with_refinement(candidate, example)
            outputs.append(output)
            scores.append(score)
            trajectories.append(side_info)
            side_infos.append(side_info)
            obj_scores = dict(side_info.get("scores", {}) or {})
            refiner_scores = side_info.get("refiner_prompt_specific_info", {}).get("scores", {})
            for key, value in dict(refiner_scores or {}).items():
                obj_scores[f"refiner_prompt::{key}"] = value
            objective_scores.append(obj_scores)
        return SimpleNamespace(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            side_infos=side_infos,
            objective_scores=objective_scores,
            num_metric_calls=len(scores),
        )
