"""LLM baseline stub — not wired. Exam 4 stays a keyword rubric.

Never emit a fabricated judge score. `scripts/run_eval.py` reports
`not_trained` even if an API key is present, until a real judge is hooked up.
"""

STATUS = "not_trained"


def explain(*args, **kwargs):
    raise NotImplementedError("LLM explain baseline is a Week-1 stub (not_trained).")


def load_model(*args, **kwargs):
    return {"status": STATUS, "model": None}
