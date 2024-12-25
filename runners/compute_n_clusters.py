import json
import time

from rdkit import Chem, SimDivFilters
from rdkit.Chem import rdMolDescriptors

from runners import setup

if __name__ == "__main__":
    args = setup.add_run_arguments(
        [
            "--sampling-strategy",
            "--t",
            "--k",
            "--p",
            "--model-name",
            "--dataset-name",
            "--evaluation-resolution",
        ]
    )

    strategy = args.sampling_strategy
    temperature = args.t
    top_k = args.k
    top_p = args.p
    model_name = args.model_name
    dataset_name = args.dataset_name
    resolution = args.evaluation_resolution

    print(f"Sampling strategy: {strategy}")
    print(f"Temperature: {temperature}")
    print(f"Top k: {top_k}")
    print(f"Top p: {top_p}")
    print(f"Model name: {model_name}")
    print(f"Dataset name: {dataset_name}")
    print(f"Evaluation resolution: {resolution}")

    DISTANCE_THRESHOLD = 0.6

    for setup_idx in range(5):
        print(f"Setup: {setup_idx}")
        designs_dir = f"./designs/{dataset_name}/setup-{setup_idx}/{model_name}/{strategy}/t={temperature}-k={top_k}-p={top_p}/"
        with open(f"{designs_dir}/valid_unique_novel_designs.smiles", "r") as f:
            designs = [line.strip() for line in f.readlines()]

        n_batches = len(designs) // 1000 + 1
        bit_vects = list()
        scores = dict()
        pool_sizes = list(range(1000, 10000, 1000))
        pool_sizes.extend(range(10000, 100000, 10000))
        pool_sizes.extend(range(100000, len(designs), 100000))
        pool_sizes.append(len(designs))
        for pool_idx, pool_size in enumerate(pool_sizes):
            prev_pool_size = pool_sizes[pool_idx - 1] if pool_idx > 0 else 0
            designs_batch = designs[prev_pool_size:pool_size]
            molecule_batch = [
                Chem.MolFromSmiles(line.strip()) for line in designs_batch
            ]
            bit_vect_batch = [
                rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                for mol in molecule_batch
            ]
            del molecule_batch, designs_batch
            bit_vects.extend(bit_vect_batch)
            assert len(bit_vects) == pool_size

            lp = SimDivFilters.LeaderPicker()
            start = time.time()
            picks = lp.LazyBitVectorPick(bit_vects, len(bit_vects), DISTANCE_THRESHOLD)
            end = time.time()
            print(
                f" {dataset_name} setup-{setup_idx} - picking took {end - start:.2f} sec for pool size {len(bit_vects)}. n_leads: {len(picks)}"
            )

            scores[f"@{len(bit_vects)}"] = len(picks)
            scores[f"@{len(bit_vects)}_time"] = end - start

            with open(f"{designs_dir}/n_leads.json", "w") as f:
                json.dump(scores, f, indent=4)
