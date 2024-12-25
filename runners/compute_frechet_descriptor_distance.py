# %%
import json
import time

import numpy as np
from scipy import linalg


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Borrowed from the FCD library"""

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    is_real = np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3)

    if not np.isfinite(covmean).all() or not is_real:
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    assert isinstance(covmean, np.ndarray)
    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def scale_columns(X):
    return (X - descriptor_mins) / descriptor_ranges


descriptor_names = [
    "logp",
    "mws",
    "n_hb_donors",
    "n_rings",
    "tpsa",
]
descriptor_mins = np.array([-3, 0, 0, 0, 0])
descriptor_maxes = np.array([10, 1000, 10, 10, 250])
descriptor_ranges = descriptor_maxes - descriptor_mins


pool_sizes = list(range(100, 1000, 100))
pool_sizes.extend(range(1000, 10000, 1000))
pool_sizes.extend(range(10000, 100000, 10000))
pool_sizes.extend(range(100000, 1000001, 100000))

design_model_name = ["lstm", "s4", "gpt"]
for model_name in design_model_name:
    for dataset in ["DRD3", "PIN1", "VDR"]:
        for setup_idx in range(5):
            start = time.time()
            print(f"Computing fdd for {dataset} setup-{setup_idx}")

            train_descriptor_values = list()
            for descriptor_name in descriptor_names:
                with open(
                    f"data/{dataset}/setup-{setup_idx}/train_descriptors/{descriptor_name}.txt",
                    "r",
                ) as f:
                    descriptor_values = [float(line.strip()) for line in f.readlines()]
                train_descriptor_values.append(descriptor_values)

            train_descriptor_values = np.array(train_descriptor_values).T
            train_descriptor_values = scale_columns(train_descriptor_values)

            mu_train = np.mean(train_descriptor_values, axis=0)
            sigma_train = np.cov(train_descriptor_values.T)

            test_descriptor_values = list()
            for descriptor_name in descriptor_names:
                with open(
                    f"data/{dataset}/setup-{setup_idx}/test_descriptors/{descriptor_name}.txt",
                    "r",
                ) as f:
                    descriptor_values = [float(line.strip()) for line in f.readlines()]
                test_descriptor_values.append(descriptor_values)

            test_descriptor_values = np.array(test_descriptor_values).T
            test_descriptor_values = scale_columns(test_descriptor_values)

            mu_test = np.mean(test_descriptor_values, axis=0)
            sigma_test = np.cov(test_descriptor_values.T)

            design_descriptor_values = list()
            for descriptor_name in descriptor_names:
                with open(
                    f"designs/{dataset}/setup-{setup_idx}/{model_name}/temperature/t=1.0-k=33-p=1.0/descriptors/{descriptor_name}.txt",
                    "r",
                ) as f:
                    descriptor_values = [float(line.strip()) for line in f.readlines()]
                design_descriptor_values.append(descriptor_values)

            design_descriptor_values = np.array(design_descriptor_values).T

            pool_size_to_fdd = dict()
            for pool_size in pool_sizes:
                if pool_size > len(design_descriptor_values):
                    break
                pool_descriptors = design_descriptor_values[:pool_size, :]
                pool_descriptors = scale_columns(pool_descriptors)
                mu_pool = np.mean(pool_descriptors, axis=0)
                sigma_pool = np.cov(pool_descriptors.T)

                pool_fdd_train = calculate_frechet_distance(
                    mu1=mu_train, mu2=mu_pool, sigma1=sigma_train, sigma2=sigma_pool
                )
                pool_fdd_test = calculate_frechet_distance(
                    mu1=mu_test, mu2=mu_pool, sigma1=sigma_test, sigma2=sigma_pool
                )

                pool_size_to_fdd[pool_size] = {
                    "train": pool_fdd_train,
                    "test": pool_fdd_test,
                }

                with open(
                    f"designs/{dataset}/setup-{setup_idx}/{model_name}/temperature/t=1.0-k=33-p=1.0/fdd_distances.json",
                    "w",
                ) as f:
                    json.dump(pool_size_to_fdd, f, indent=4)

                print(f"Pool size: {pool_size}")
                print(f"FDD: {pool_size_to_fdd[pool_size]}")

            end = time.time()
            print(f"Time taken: {(end - start) // 60} minutes")
            print("=====================================")
