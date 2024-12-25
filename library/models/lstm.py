from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from library.data import data_utils
from library.models.ar_clm import AutoRegressiveChemicalLanguageModel


class LSTM(AutoRegressiveChemicalLanguageModel):
    def __init__(
        self,
        **shared_clm_args,
    ):
        super().__init__(**shared_clm_args)

    def build_architecture(self):
        return nn.ModuleDict(
            dict(
                embedding=nn.Embedding(self.vocab_size, self.model_dim, padding_idx=0),
                lstm=nn.LSTM(
                    self.model_dim,
                    self.model_dim,
                    self.n_layers,
                    batch_first=True,
                    dropout=self.dropout,
                ),
                lm_head=nn.Linear(self.model_dim, self.vocab_size),
            )
        )

    def forward(
        self,
        x: torch.LongTensor,
        hidden_states: torch.FloatTensor = None,
        training: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if len(x.shape) == 1:
            x = x.unsqueeze(1)  # (batch_size, 1, sequence_length)
        x = self.architecture.embedding(x)  # (batch_size, sequence_length, model_dim)
        x, hidden_states = self.architecture.lstm(
            x, hidden_states
        )  # (batch_size, sequence_length, model_dim)
        x = self.architecture.lm_head(x)  # (batch_size, sequence_length, vocab_size)
        if training:
            return x  # non-normalized logits (batch_size, sequence_length, vocab_size)

        return x, hidden_states

    def initialize_hidden_states(self, batch_size: int):
        return (
            torch.zeros(self.n_layers, batch_size, self.model_dim)
            .float()
            .to(self.device),
            torch.zeros(self.n_layers, batch_size, self.model_dim)
            .float()
            .to(self.device),
        )

    @torch.no_grad()
    def compute_log_likelihood_of_molecules(
        self, smiles_batch: str, batch_size: int, token2label: Dict[str, int]
    ) -> List[float]:
        self = self.to(self.device)
        self.eval()
        smiles_tensor = data_utils.molecules_to_tensor(
            smiles_batch, self.sequence_length, token2label
        )[:, :-1].to(self.device)
        log_likelihoods = list()
        for batch_idx in range(0, smiles_tensor.shape[0], batch_size):
            batch_smiles_tensor = smiles_tensor[batch_idx : batch_idx + batch_size, :]
            batch_preds = self.forward(batch_smiles_tensor, hidden_states=None)
            log_preds = F.log_softmax(batch_preds, -1)
            idxes_to_fetch = batch_smiles_tensor[:, 1:].unsqueeze(2)
            batch_log_likelihoods = torch.gather(log_preds, 2, idxes_to_fetch)
            batch_log_likelihoods = (
                batch_log_likelihoods.squeeze(2).cpu().numpy().tolist()
            )
            log_likelihoods.extend(batch_log_likelihoods)

        label_encoded_smiles = smiles_tensor.cpu().numpy().tolist()
        end_idx = token2label[data_utils.end_token]
        try:
            end_idxes = [smiles.index(end_idx) for smiles in label_encoded_smiles]
        except ValueError:
            for smiles in label_encoded_smiles:
                try:
                    smiles.index(end_idx)
                except ValueError:
                    print(smiles)
            raise ValueError("End token not found in the label encoded smiles.")
        log_likelihoods = [
            np.mean(log_likelihood[:idx]).astype(float)
            for log_likelihood, idx in zip(log_likelihoods, end_idxes)
        ]
        return log_likelihoods
