from gepa.api import optimize
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    MergeConfig,
    TrackingConfig,
    _serialize_config,
)
from gepa.strategies.instruction_proposal import InstructionProposalSignature


class ProtocolDefaultAdapterLike:
    propose_new_texts = None

    def __init__(self):
        self.evaluate_calls = 0
        self.reflect_calls = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        self.evaluate_calls += 1
        return {
            "outputs": [{"answer": "ok"} for _ in batch],
            "scores": [1.0 for _ in batch],
            "trajectories": [{"Feedback": "ok"} for _ in batch] if capture_traces else None,
            "objective_scores": [{"score": 1.0} for _ in batch],
            "num_metric_calls": len(batch),
        }

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        self.reflect_calls += 1
        return {name: [{"Feedback": "ok"}] for name in components_to_update}


class CallableProposerAdapter:
    def __init__(self):
        self.evaluate_calls = 0
        self.reflect_calls = 0
        self.propose_calls = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        self.evaluate_calls += 1
        return {
            "outputs": [{"answer": "ok"} for _ in batch],
            "scores": [1.0 for _ in batch],
            "trajectories": [{"Feedback": "ok"} for _ in batch] if capture_traces else None,
            "objective_scores": [{"score": 1.0} for _ in batch],
            "num_metric_calls": len(batch),
        }

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        self.reflect_calls += 1
        return {name: [{"Feedback": "ok"}] for name in components_to_update}

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        self.propose_calls += 1
        return {"system_prompt": "improved"}


class PromptCaptureAdapter(ProtocolDefaultAdapterLike):
    def evaluate(self, batch, candidate, capture_traces=False):
        self.evaluate_calls += 1
        return {
            "outputs": [{"answer": "ok"} for _ in batch],
            "scores": [0.25 for _ in batch],
            "trajectories": [{"Feedback": "needs work"} for _ in batch] if capture_traces else None,
            "objective_scores": [{"score": 0.25} for _ in batch],
            "num_metric_calls": len(batch),
        }


def test_protocol_none_proposer_adapter_evaluates_without_custom_proposer():
    adapter = ProtocolDefaultAdapterLike()

    result = optimize(
        seed_candidate={"system_prompt": "seed"},
        trainset=[{"input": "train", "answer": "ok", "additional_context": {}}],
        valset=[{"input": "val", "answer": "ok", "additional_context": {}}],
        adapter=adapter,
        reflection_lm=lambda prompt: "```\nimproved\n```",
        max_metric_calls=2,
        display_progress_bar=False,
    )

    assert adapter.evaluate_calls > 0
    assert result.best_candidate["system_prompt"] == "seed"


def test_callable_proposer_adapter_is_forwarded_to_sidecar():
    adapter = CallableProposerAdapter()

    result = optimize(
        seed_candidate={"system_prompt": "seed"},
        trainset=[{"input": "train", "answer": "ok", "additional_context": {}}],
        valset=[{"input": "val", "answer": "ok", "additional_context": {}}],
        adapter=adapter,
        reflection_lm=lambda prompt: "```\nunused\n```",
        max_metric_calls=3,
        display_progress_bar=False,
    )

    assert adapter.evaluate_calls > 0
    assert result.best_candidate["system_prompt"] in {"seed", "improved"}


def test_engine_frontier_type_is_serialized_for_sidecar():
    config = GEPAConfig(engine=EngineConfig(frontier_type="objective"))

    serialized = _serialize_config(config)

    assert serialized["engine"]["frontier_type"] == "objective"


def test_merge_config_is_serialized_for_sidecar():
    config = GEPAConfig(
        merge=MergeConfig(max_merge_invocations=2, merge_val_overlap_floor=3)
    )

    serialized = _serialize_config(config)

    assert serialized["merge"] == {
        "max_merge_invocations": 2,
        "merge_val_overlap_floor": 3,
    }


def test_tracking_config_is_serialized_for_sidecar():
    config = GEPAConfig(
        tracking=TrackingConfig(
            use_wandb=True,
            wandb_api_key="key",
            wandb_init_kwargs={"project": "gepa"},
            wandb_attach_existing=True,
            wandb_step_metric="iteration",
            use_mlflow=True,
            mlflow_tracking_uri="file:///tmp/mlruns",
            mlflow_experiment_name="exp",
            mlflow_attach_existing=True,
            key_prefix="gepa/",
        )
    )

    serialized = _serialize_config(config)

    assert serialized["tracking"] == {
        "use_wandb": True,
        "wandb_api_key": "key",
        "wandb_init_kwargs": {"project": "gepa"},
        "wandb_attach_existing": True,
        "wandb_step_metric": "iteration",
        "use_mlflow": True,
        "mlflow_tracking_uri": "file:///tmp/mlruns",
        "mlflow_experiment_name": "exp",
        "mlflow_attach_existing": True,
        "key_prefix": "gepa/",
    }


def test_optimize_default_reflection_prompt_matches_upstream_instruction_default():
    adapter = PromptCaptureAdapter()
    prompts = []

    def reflection_lm(prompt):
        prompts.append(prompt)
        return "```\nimproved\n```"

    optimize(
        seed_candidate={"system_prompt": "seed"},
        trainset=[
            {"input": "train-1", "answer": "ok", "additional_context": {}},
            {"input": "train-2", "answer": "ok", "additional_context": {}},
            {"input": "train-3", "answer": "ok", "additional_context": {}},
        ],
        valset=[{"input": "val", "answer": "ok", "additional_context": {}}],
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=5,
        display_progress_bar=False,
    )

    assert prompts
    assert prompts[0].startswith(
        "I provided an assistant with the following instructions to perform a task for me:"
    )
    assert not prompts[0].startswith("I am optimizing a parameter in my system.")
    assert prompts[0].split("```", 1)[0] == InstructionProposalSignature.default_prompt_template.split("```", 1)[0]
