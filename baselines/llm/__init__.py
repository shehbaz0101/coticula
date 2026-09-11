"""LLM baseline stub — not trained / not wired in Week-1 scaffold."""

STATUS = "not_trained"


def explain(*args, **kwargs):
    raise NotImplementedError("LLM explain baseline is a Week-1 stub (not_trained).")


def load_model(*args, **kwargs):
    return {"status": STATUS, "model": None}
