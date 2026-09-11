"""PINO baseline stub — not trained in Week-1 scaffold."""

STATUS = "not_trained"


def predict(*args, **kwargs):
    raise NotImplementedError("PINO baseline is a Week-1 stub (not_trained).")


def load_model(*args, **kwargs):
    return {"status": STATUS, "model": None}
