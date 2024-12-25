# %%
import json

from rdkit import Chem
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
            "--use-synthesizable-designs",
        ]
    )

    strategy = args.sampling_strategy
    temperature = args.t
    top_k = args.k
    top_p = args.p
    model_name = args.model_name
    dataset_name = args.dataset_name
    resolution = args.evaluation_resolution
    synthesizable_designs = args.use_synthesizable_designs

    print(f"Sampling strategy: {strategy}")
    print(f"Temperature: {temperature}")
    print(f"Top k: {top_k}")
    print(f"Top p: {top_p}")
    print(f"Model name: {model_name}")
    print(f"Dataset name: {dataset_name}")
    print(f"Evaluation resolution: {resolution}")

    for setup_idx in range(5):
        print(f"Setup: {setup_idx}")
        designs_dir = f"./designs/{dataset_name}/setup-{setup_idx}/{model_name}/{strategy}/t={temperature}-k={top_k}-p={top_p}/"

        if synthesizable_designs == "True":
            with open(f"{designs_dir}/low_sa_designs.smiles", "r") as f:
                designs = [line.strip() for line in f.readlines()]
        elif synthesizable_designs == "False":
            with open(f"{designs_dir}/valid_unique_novel_designs.smiles", "r") as f:
                designs = [line.strip() for line in f.readlines()]

        n_batches = len(designs) // 1000 + 1
        n_unique_substructures, substructures = dict(), dict()
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
            morgan_vects_batch = [
                rdMolDescriptors.GetMorganFingerprint(mol, radius=2)
                for mol in molecule_batch
            ]
            non_zeros_batch = [fp.GetNonzeroElements() for fp in morgan_vects_batch]
            for non_zeros in non_zeros_batch:
                for key, value in non_zeros.items():
                    if key not in substructures:
                        substructures[key] = value
                    else:
                        substructures[key] += value

            n_unique_substructures[pool_size] = len(substructures)

            del molecule_batch, designs_batch, morgan_vects_batch, non_zeros_batch
            print(
                f"Pool size: {pool_size}, n_unique_substructures: {n_unique_substructures[pool_size]}"
            )

            fname = (
                "n_unique_substructures.json"
                if synthesizable_designs == "False"
                else "n_unique_substructures_low_sa.json"
            )
            with open(f"{designs_dir}/{fname}", "w") as f:
                json.dump(n_unique_substructures, f, indent=4)

        fname = (
            "substructures.json"
            if synthesizable_designs == "False"
            else "substructures_low_sa.json"
        )
        with open(f"{designs_dir}/{fname}", "w") as f:
            json.dump(substructures, f, indent=4)
