from .gpt import GPT  # noqa: F401
from .lstm import LSTM  # noqa: F401
from .s4 import S4  # noqa: F401

__MODEL_NAMES__ = {
    "gpt": GPT,
    "lstm": LSTM,
    "s4": S4,
}


def get_chemical_language_model(model_name: str):
    clm = __MODEL_NAMES__.get(model_name, None)
    if clm is None:
        raise ValueError(f"Unknown model name {model_name}")
    return clm
