# %%

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

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

    for setup_idx in range(5):
        n_errors = 0
        print(f"Setup: {setup_idx}")
        designs_dir = f"./designs/{dataset_name}/setup-{setup_idx}/{model_name}/{strategy}/t={temperature}-k={top_k}-p={top_p}/"
        with open(f"{designs_dir}/valid_unique_novel_designs.smiles", "r") as f:
            designs = [line.strip() for line in f.readlines()]

        scaffold_smiles, generic_scaffold_smiles = list(), list()
        for design in designs:
            try:
                mol = Chem.MolFromSmiles(design)
                scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles.append(Chem.MolToSmiles(scaffold_mol))
            except Exception:
                n_errors += 1
                scaffold_smiles.append("ERROR")
                print(f"Error with design: {design}")

            try:
                generic_scaffold_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
                generic_scaffold_smiles.append(Chem.MolToSmiles(generic_scaffold_mol))
            except Exception:
                generic_scaffold_smiles.append("ERROR")
                n_errors += 1
                print(f"Error with design: {design}")

        with open(f"{designs_dir}/scaffolds.smiles", "w") as f:
            f.write("\n".join(scaffold_smiles))

        with open(f"{designs_dir}/generic_scaffolds.smiles", "w") as f:
            f.write("\n".join(generic_scaffold_smiles))

        print(f"Number of errors: {n_errors}")
