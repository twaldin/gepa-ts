from gepa.optimize_anything import (
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
    optimize_anything,
)


def optimize(
    seed_candidate,
    *,
    trainset,
    task_lm=None,
    reflection_lm=None,
    max_metric_calls=None,
    reflection_minibatch_size=None,
    reflection_prompt_template=None,
    callbacks=None,
    **kwargs,
):
    if seed_candidate is None or (
        isinstance(seed_candidate, dict) and not seed_candidate
    ):
        raise ValueError("seed_candidate must contain at least one component text.")

    if isinstance(reflection_prompt_template, str):
        missing = [
            p
            for p in ["<curr_param>", "<side_info>"]
            if p not in reflection_prompt_template
        ]
        if missing:
            raise ValueError(
                f"Missing placeholder(s) in prompt template: {', '.join(missing)}"
            )
    elif isinstance(reflection_prompt_template, dict):
        for param_name, template in reflection_prompt_template.items():
            if not isinstance(template, str):
                raise ValueError(
                    f"reflection_prompt_template['{param_name}'] must be a string"
                )
            missing = [
                p for p in ["<curr_param>", "<side_info>"] if p not in template
            ]
            if missing:
                raise ValueError(
                    f"Missing placeholder(s) in prompt template for parameter '{param_name}': {', '.join(missing)}"
                )

    def _evaluator(candidate, example):
        si: dict = {}
        if isinstance(example, dict):
            for k, v in example.items():
                if isinstance(v, (str, int, float)):
                    si[str(k)] = str(v)
                elif isinstance(v, dict):
                    si[str(k)] = str(v)
        prompt = str(example)
        try:
            response_val = task_lm(prompt) if task_lm is not None else "response"
        except Exception:
            response_val = "response"
        si["task_output"] = str(response_val)
        return 0.0, si

    engine = EngineConfig(max_metric_calls=max_metric_calls)
    reflection = ReflectionConfig(
        reflection_lm=reflection_lm,
        reflection_minibatch_size=reflection_minibatch_size,
        reflection_prompt_template=reflection_prompt_template,
    )
    config = GEPAConfig(engine=engine, reflection=reflection, callbacks=callbacks)

    return optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=_evaluator,
        dataset=trainset,
        config=config,
    )
