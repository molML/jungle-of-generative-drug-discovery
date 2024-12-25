import json
import os
import time

import torch

from library.models import get_chemical_language_model
from runners.setup import add_run_arguments

if __name__ == "__main__":
    args = add_run_arguments(
        [
            "--sampling-strategy",
            "--t",
            "--k",
            "--p",
            "--model-name",
            "--dataset-name",
        ]
    )

    strategy = args.sampling_strategy
    temperature = args.t
    top_k = args.k
    top_p = args.p
    model_name = args.model_name
    dataset_name = args.dataset_name

    print(f"Sampling strategy: {strategy}")
    print(f"Temperature: {temperature}")
    print(f"Top k: {top_k}")
    print(f"Top p: {top_p}")
    print(f"Model name: {model_name}")
    print(f"Dataset name: {dataset_name}")

    if model_name == "gpt":
        BATCH_SIZE = 2**16
        N_BATCHES = 2**4
    elif model_name == "lstm":
        BATCH_SIZE = 2**18
        N_BATCHES = 2**2
    elif model_name == "s4":
        BATCH_SIZE = 2**12
        N_BATCHES = 2**8 - 11

    with open("data/token2label.json", "r") as f:
        token2label = json.load(f)

    for setup_idx in range(5):
        clm = get_chemical_language_model(model_name).from_checkpoint(
            f"./models/{dataset_name}/setup-{setup_idx}/{model_name}/last-epoch/"
        )
        saving_dir = f"./designs/{dataset_name}/setup-{setup_idx}/{model_name}/{strategy}/t={temperature}-k={top_k}-p={top_p}/"

        os.makedirs(saving_dir, exist_ok=True)
        with open(f"{saving_dir}/designs.txt", "w") as f:
            f.write("")

        with open(f"{saving_dir}/lls.txt", "w") as f:
            f.write("")

        start = time.time()
        for batch_idx in range(N_BATCHES):
            batch_start = time.time()
            torch.manual_seed(batch_idx)
            designs, loglikelihoods = clm.design_molecules(
                n_batches=1,
                batch_size=BATCH_SIZE,
                temperature=temperature,
                token2label=token2label,
                top_k=top_k,
                top_p=top_p,
            )

            with open(f"{saving_dir}/designs.txt", "a") as f:
                f.write("\n".join(designs))
                f.write("\n")

            with open(f"{saving_dir}/lls.txt", "a") as f:
                f.write("\n".join([str(elem) for elem in loglikelihoods]))
                f.write("\n")

            del designs, loglikelihoods
            batch_end = time.time()
            print(f"Batch {batch_idx} took {batch_end - batch_start:.2f} seconds.")

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds.")
