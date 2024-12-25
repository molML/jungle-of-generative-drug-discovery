# %%
from typing import Dict, Tuple

import torch

from library.data import data_utils


class CLMLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        label_encoded_molecules: torch.LongTensor,
    ):
        self.label_encoded_molecules = label_encoded_molecules

    def __len__(self) -> int:
        return self.label_encoded_molecules.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.LongTensor, torch.LongTensor]:
        molecule = self.label_encoded_molecules[idx, :]
        X = molecule[:-1]
        y = molecule[1:]
        return X, y


def get_dataloader(
    path_to_data: str,
    batch_size: int,
    sequence_length: int,
    token2label: Dict[str, int],
    n_samples: int = None,
    num_workers: int = 8,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    with open(path_to_data, "r") as f:
        molecules = [line.strip() for line in f.readlines()]

    if n_samples is not None:
        molecules = molecules[:n_samples]
        batch_size = 4

    molecules_tensor = data_utils.molecules_to_tensor(
        molecules, sequence_length, token2label, space_separated=True
    )

    return torch.utils.data.DataLoader(
        CLMLoader(
            molecules_tensor,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    import json

    with open("./data/token2label.json", "r") as f:
        token2label = json.load(f)

    train_loader = get_dataloader(
        "./datasets/chemblv31/train.smiles",
        batch_size=16,
        sequence_length=100,
        token2label=token2label,
        num_workers=8,
        shuffle=True,
    )
    val_loader = get_dataloader(
        "./datasets/chemblv31/valid.smiles",
        batch_size=16,
        sequence_length=100,
        token2label=token2label,
        num_workers=8,
        shuffle=False,
    )
