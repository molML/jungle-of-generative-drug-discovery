import os
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from library.smiles import sascorer, smiles_utils
from runners import cats, setup


def max_tanimoto_to_train(mol, training_fps, fp_generator):
    mol_fp = fp_generator.GetFingerprint(mol)
    return max([Chem.DataStructs.TanimotoSimilarity(mol_fp, fp) for fp in training_fps])


DESCRIPTOR_NAMES = [
    "mws",
    "logp",
    "n_rings",
    "tpsa",
    "n_hb_donors",
    "sas",
    "tanimoto_to_train",
]
DESCRIPTOR_TO_FN = {
    "mws": Descriptors.MolWt,
    "logp": Descriptors.MolLogP,
    "qed": Descriptors.qed,
    "n_rings": Descriptors.RingCount,
    "tpsa": Descriptors.TPSA,
    "n_hb_donors": Descriptors.NumHDonors,
    "sas": sascorer.calculateScore,
    "tanimoto_to_train": max_tanimoto_to_train,
}


def compute_descriptors(
    designs: List[str],
    saving_dir: str,
    training_set: List[str],
    descriptors_to_compute: List[str],
):
    all_values = {descriptor_name: list() for descriptor_name in DESCRIPTOR_NAMES}
    reference_mols = [Chem.MolFromSmiles(smiles) for smiles in training_set]
    fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    reference_fps = [fpgen.GetFingerprint(m) for m in reference_mols]
    training_lens = [len(smiles_utils.segment_smiles(smi)) for smi in training_set]
    for design_idx, design in enumerate(designs):
        if design_idx % 10000 == 0:
            print(f"{design_idx}/{len(designs)} designs computed")
        mol = Chem.MolFromSmiles(design)
        for descriptor_name in descriptors_to_compute:
            if descriptor_name == "ed_to_train":
                descriptor_value = DESCRIPTOR_TO_FN[descriptor_name](
                    design, training_set, training_lens
                )
            elif descriptor_name == "smiles_len":
                descriptor_value = DESCRIPTOR_TO_FN[descriptor_name](design)
            elif descriptor_name == "tanimoto_to_train":
                descriptor_value = DESCRIPTOR_TO_FN[descriptor_name](
                    mol, reference_fps, fp_generator=fpgen
                )
            elif descriptor_name == "cats_to_train":
                reference_cats = [cats.one_cats(mol) for mol in reference_mols]
                descriptor_value = DESCRIPTOR_TO_FN[descriptor_name](
                    mol, reference_cats
                )
            else:
                descriptor_value = DESCRIPTOR_TO_FN[descriptor_name](mol)

            all_values[descriptor_name].append(descriptor_value)

    for name, values in all_values.items():
        with open(f"{saving_dir}/{name}.txt", "w") as f:
            f.write("\n".join([str(d) for d in values]))


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
        print(f"Setup: {setup_idx}")
        with open(f"data/{dataset_name}/setup-{setup_idx}/train.smiles", "r") as f:
            training_set = [line.strip() for line in f.readlines()]

        training_set = [smiles.replace(" ", "") for smiles in training_set]
        designs_dir = f"./designs/{dataset_name}/setup-{setup_idx}/{model_name}/{strategy}/t={temperature}-k={top_k}-p={top_p}/"

        with open(f"{designs_dir}/valid_unique_novel_designs.smiles", "r") as f:
            designs = [line.strip() for line in f.readlines()]

        saving_dir = f"{designs_dir}/descriptors/"
        os.makedirs(saving_dir, exist_ok=True)

        descriptors_to_compute = list()
        for descriptor_name in DESCRIPTOR_NAMES:
            if os.path.exists(f"{saving_dir}/{descriptor_name}.txt"):
                with open(f"{saving_dir}/{descriptor_name}.txt", "r") as f:
                    lines = f.readlines()

                if len(lines) == len(designs):
                    print("Descriptors already computed")
                else:
                    descriptors_to_compute.append(descriptor_name)
            else:
                descriptors_to_compute.append(descriptor_name)

        if len(descriptors_to_compute) > 0:
            compute_descriptors(
                designs,
                saving_dir=saving_dir,
                training_set=training_set,
                descriptors_to_compute=descriptors_to_compute,
            )
