from typing import Dict, List

import numpy as np
import torch
from torch import nn

from library.data import data_utils
from library.models.ar_clm import AutoRegressiveChemicalLanguageModel
from library.models.s4_modules.sequence_model import SequenceModel


class S4(AutoRegressiveChemicalLanguageModel):
    def __init__(
        self,
        state_dim: int,
        n_ssm: int,
        **shared_clm_args,
    ) -> None:
        self.state_dim = state_dim
        self.n_ssm = n_ssm
        super().__init__(**shared_clm_args)

    def build_architecture(self):
        layer_config = [
            {
                "_name_": "s4",
                "d_state": self.state_dim,
                "n_ssm": self.n_ssm,
            },
            {
                "_name_": "s4",
                "d_state": self.state_dim,
                "n_ssm": self.n_ssm,
            },
            {"_name_": "ff"},
        ]
        pool_config = {"_name_": "pool", "stride": 1, "expand": None}

        return nn.ModuleDict(
            dict(
                embedding=nn.Embedding(
                    self.vocab_size, self.model_dim
                ),  # cannot use padding_idx=0. Raises nan values
                s4=SequenceModel(
                    d_model=self.model_dim,
                    n_layers=self.n_layers,
                    transposed=True,
                    dropout=self.dropout,
                    layer=layer_config,
                    pool=pool_config,
                ),
                lm_head=nn.Linear(self.model_dim, self.vocab_size),
            )
        )

    def forward(
        self,
        x: torch.LongTensor,
        hidden_states: torch.FloatTensor = None,
        training: bool = True,
    ) -> torch.Tensor:
        x = self.architecture.embedding(x)
        if training:
            x, hidden_states = self.architecture.s4(
                x, state=hidden_states
            )  # hidden_states are None in convolutional form
        else:
            x, hidden_states = self.architecture.s4.step(x, state=hidden_states)
        x = self.architecture.lm_head(x)

        if training:
            return x
        return x, hidden_states

    def initialize_hidden_states(self, batch_size: int) -> torch.Tensor:
        return self.architecture.s4.default_state(batch_size, device=self.device)

    @torch.no_grad()
    def compute_log_likelihood_of_molecules(
        self, smiles_batch: str, batch_size: int, token2label: Dict[str, int]
    ) -> List[float]:
        smiles_tensor = data_utils.molecules_to_tensor(
            smiles_batch, self.sequence_length, token2label
        )[:, :-1].to(self.device)
        for module in self.architecture.s4.modules():
            if hasattr(module, "setup_step"):
                module.setup_step()
        self = self.to(self.device)
        self.eval()
        log_likelihoods = list()
        for batch_idx in range(0, smiles_tensor.shape[0], batch_size):
            batch_smiles_tensor = smiles_tensor[batch_idx : batch_idx + batch_size, :]
            hidden_states = self.initialize_hidden_states(batch_smiles_tensor.shape[0])
            batch_log_likelihoods = list()
            for token_idx in range(self.sequence_length - 1):
                token_preds, hidden_states = self.forward(
                    batch_smiles_tensor[:, token_idx],
                    hidden_states=hidden_states,
                    training=False,
                )
                token_log_likelihoods = torch.gather(
                    torch.log_softmax(token_preds, -1),
                    1,
                    batch_smiles_tensor[:, token_idx + 1].unsqueeze(1),
                )
                batch_log_likelihoods.append(token_log_likelihoods.squeeze(1).tolist())

            log_likelihoods.extend(np.array(batch_log_likelihoods).T.tolist())

        label_encoded_smiles = smiles_tensor.cpu().numpy().tolist()
        end_idx = token2label[data_utils.end_token]
        end_idxes = [smiles.index(end_idx) for smiles in label_encoded_smiles]
        log_likelihoods = [
            np.mean(log_likelihood[:idx]).astype(float)
            for log_likelihood, idx in zip(log_likelihoods, end_idxes)
        ]
        return log_likelihoods
