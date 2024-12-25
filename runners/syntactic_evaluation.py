import json

from library.smiles import smiles_utils
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

    with open("data/chemblv33/train.smiles", "r") as f:
        chembl = [line.strip() for line in f.readlines()]

    chembl = set([smiles.replace(" ", "") for smiles in chembl])
    for setup_idx in range(5):
        print(f"Setup: {setup_idx}")
        with open(f"data/{dataset_name}/setup-{setup_idx}/train.smiles", "r") as f:
            training_dataset = [line.strip() for line in f.readlines()]
            training_dataset = [smiles.replace(" ", "") for smiles in training_dataset]
            training_dataset = set(training_dataset) | chembl

        designs_dir = f"./designs/{dataset_name}/setup-{setup_idx}/{model_name}/{strategy}/t={temperature}-k={top_k}-p={top_p}/"

        with open(f"{designs_dir}/designs.txt", "r") as f:
            designs = [line.strip() for line in f.readlines()]

        print(len(designs))
        if len(designs) < 10**6:
            raise ValueError(
                f"Number of designs ({len(designs)}) does not match expected number of designs (2^20)"
            )
        designs = designs[: 10**6]
        syntactic_scores = dict()
        valid_designs = list()
        unique_designs = set()
        novel_designs = set()
        n_batches = len(designs) // resolution
        for batch_idx in range(n_batches):
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{n_batches}")
            batch_start_idx = batch_idx * resolution
            batch_end_idx = (batch_idx + 1) * resolution
            design_batch = designs[batch_start_idx:batch_end_idx]

            valid_design_batch = smiles_utils.sanitize_smiles_batch(
                design_batch, to_canonical=True
            )
            valid_design_batch = [
                design
                for design in valid_design_batch
                if design is not None and len(design) > 0
            ]
            valid_designs.extend(valid_design_batch)

            unique_designs_batch = set(valid_design_batch)
            unique_designs.update(unique_designs_batch)

            novel_designs_batch = unique_designs_batch - training_dataset
            novel_designs.update(novel_designs_batch)

            n_generations = (batch_idx + 1) * resolution
            validity_at_k = len(valid_designs) / n_generations
            uniqueness_at_k = len(unique_designs) / n_generations
            novelty_at_k = len(novel_designs) / n_generations

            syntactic_scores[f"@{n_generations}"] = {
                "validity": validity_at_k,
                "uniqueness": uniqueness_at_k,
                "novelty": novelty_at_k,
                "n_valid": len(valid_designs),
                "n_unique": len(unique_designs),
                "n_novel": len(novel_designs),
            }

            with open(f"{designs_dir}/syntactic_scores.json", "w") as f:
                json.dump(syntactic_scores, f, indent=4)

            with open(f"{designs_dir}/valid_unique_novel_designs.smiles", "w") as f:
                f.write("\n".join(novel_designs))
