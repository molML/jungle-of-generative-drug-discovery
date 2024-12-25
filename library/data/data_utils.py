from typing import Dict, List, Union

import torch

from library.data import data_utils
from library.smiles import smiles_utils

__BEG_TOKEN = "<BEG>"
__END_TOKEN = "<END>"
__PAD_TOKEN = "<PAD>"

__ATTRIBUTES = {
    "beg_token": __BEG_TOKEN,
    "end_token": __END_TOKEN,
    "pad_token": __PAD_TOKEN,
}


def __getattr__(name):
    if name in __ATTRIBUTES:
        return __ATTRIBUTES[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def pad_sequences(
    sequences: List[List[Union[str, int]]],
    padding_length: int,
    padding_value: Union[str, int],
) -> List[List[Union[str, int]]]:
    lens = [len(seq) for seq in sequences]
    diffs = [max(padding_length - len, 0) for len in lens]
    padded_sequences = [
        seq + [padding_value] * diff for seq, diff in zip(sequences, diffs)
    ]
    truncated_sequences = [seq[-padding_length:] for seq in padded_sequences]

    return torch.LongTensor(truncated_sequences)


def molecules_to_tensor(
    molecules: List[str],
    sequence_length: int,
    token2label: Dict[str, int],
    space_separated: bool = False,
) -> torch.LongTensor:
    if space_separated:
        tokenized_molecules = [
            [data_utils.beg_token] + smiles.strip().split() + [data_utils.end_token]
            for smiles in molecules
        ]
    else:
        tokenized_molecules = [
            [data_utils.beg_token]
            + smiles_utils.segment_smiles(smiles)
            + [data_utils.end_token]
            for smiles in molecules
        ]

    encoded_molecules = [
        [token2label[token] for token in tokens] for tokens in tokenized_molecules
    ]
    return pad_sequences(
        encoded_molecules,
        padding_length=sequence_length + 1,
        padding_value=token2label[data_utils.pad_token],
    )


def preprocess_smiles(
    smiles_batch: List[str],
    token2label: Dict[str, int] = None,
    sequence_length: int = 100,
):
    tokenized_dataset = [
        [data_utils.beg_token]
        + smiles_utils.segment_smiles(smiles)
        + [data_utils.end_token]
        for smiles in smiles_batch
    ]
    if token2label is None:
        token2label = smiles_utils.learn_label_encoding(tokenized_dataset)

    padded_dataset = smiles_utils.pad_sequences(
        tokenized_dataset, sequence_length, padding_value=data_utils.pad_token
    )
    return [[token2label[token] for token in tokens] for tokens in padded_dataset]
