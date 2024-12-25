import json
import time

import fcd
import numpy as np
from fcd.utils import calculate_frechet_distance

model = fcd.fcd.load_ref_model()

pool_sizes = list(range(100, 1000, 100))
pool_sizes.extend(range(1000, 10000, 1000))
pool_sizes.extend(range(10000, 100000, 10000))
pool_sizes.extend(range(100000, 1000001, 100000))

design_model_name = ["lstm", "s4", "gpt"]
for model_name in design_model_name:
    for dataset in ["DRD3", "PIN1", "VDR"]:
        for setup_idx in range(5):
            start = time.time()
            print(f"Computing FCD for {dataset} setup-{setup_idx}")
            with open(f"data/{dataset}/setup-{setup_idx}/train.smiles", "r") as f:
                train = [line.strip().replace(" ", "") for line in f.readlines()]

            train_activations = fcd.fcd.get_predictions(model, train)

            train_mu = np.mean(train_activations, axis=0)
            train_sigma = np.cov(train_activations.T)

            with open(f"data/{dataset}/setup-{setup_idx}/test.smiles", "r") as f:
                test = [line.strip().replace(" ", "") for line in f.readlines()]

            test_activations = fcd.fcd.get_predictions(model, test)

            test_mu = np.mean(test_activations, axis=0)
            test_sigma = np.cov(test_activations.T)

            with open(
                f"designs/{dataset}/setup-{setup_idx}/{model_name}/temperature/t=1.0-k=33-p=1.0/valid_unique_novel_designs.smiles",
                "r",
            ) as f:
                designs = [line.strip() for line in f.readlines()]

            pool_size_to_fcd = dict()
            for pool_size in pool_sizes:
                if pool_size > len(designs):
                    break
                pool_designs = designs[:pool_size]
                pool_activations = fcd.fcd.get_predictions(model, pool_designs)

                pool_mu = np.mean(pool_activations, axis=0)
                pool_sigma = np.cov(pool_activations.T)

                pool_fcd_distance_train = calculate_frechet_distance(
                    mu1=train_mu, mu2=pool_mu, sigma1=train_sigma, sigma2=pool_sigma
                )

                pool_fcd_distance_test = calculate_frechet_distance(
                    mu1=test_mu, mu2=pool_mu, sigma1=test_sigma, sigma2=pool_sigma
                )

                pool_size_to_fcd[pool_size] = {
                    "train": pool_fcd_distance_train,
                    "test": pool_fcd_distance_test,
                }

                with open(
                    f"designs/{dataset}/setup-{setup_idx}/{model_name}/temperature/t=1.0-k=33-p=1.0/fcd_distances.json",
                    "w",
                ) as f:
                    json.dump(pool_size_to_fcd, f, indent=4)

                print(f"Pool size: {pool_size}")
                print(f"FCD: {pool_size_to_fcd[pool_size]}")

            end = time.time()
            print(f"Time taken: {(end - start) // 60} minutes")
            print("=====================================")
