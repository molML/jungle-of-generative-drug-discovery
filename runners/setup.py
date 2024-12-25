import argparse
from typing import List

__ARGUMENTS = {
    "--sampling-strategy": {
        "help": "Sampling strategy to use",
        "choices": [
            "temperature",
            "topk",
            "topp",
        ],
        "type": str,
    },
    "--t": {
        "help": "Temperature value for sampling",
        "type": float,
        "required": True,
    },
    "--k": {
        "help": "Top k value for sampling",
        "type": int,
        "default": 33,
    },
    "--p": {
        "help": "Top p value for sampling",
        "type": float,
        "default": 1.0,
    },
    "--model-name": {
        "help": "Name of the model generate with",
        "choices": [
            "lstm",
            "s4",
            "gpt",
        ],
        "type": str,
    },
    "--dataset-name": {
        "help": "Name of the dataset was trained on",
        "choices": [
            "DRD3",
            "PIN1",
            "VDR",
        ],
        "type": str,
    },
}


def add_run_arguments(argument_list: List[str]):
    parser = argparse.ArgumentParser()

    for arg_name in argument_list:
        if arg_name not in __ARGUMENTS:
            raise ValueError(f"Invalid argument name: {arg_name}")
        parser.add_argument(arg_name, **__ARGUMENTS[arg_name])

    args, invalids = parser.parse_known_args()
    if len(invalids) > 0:
        raise ValueError(f"Invalid terminal arguments: {invalids}")
    return args
