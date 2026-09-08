def completion(*args, **kwargs):
    raise RuntimeError("litellm.completion was called without a test patch")


def batch_completion(*args, **kwargs):
    raise RuntimeError("litellm.batch_completion was called without a test patch")


def completion_cost(*args, **kwargs):
    return 0.0
