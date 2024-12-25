import json
from typing import Dict

import torch
from torch import nn


class ChemicalLanguageModel(nn.Module):
    def __init__(
        self,
        n_layers: int,
        model_dim: int,
        dropout: float,
        vocab_size: int,
        sequence_length: int,
        learning_rate: float,
        n_max_epochs: int,
        batch_size: int,
        device: str,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.model_dim = model_dim
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.n_max_epochs = n_max_epochs
        self.batch_size = batch_size
        self.device = device

        self.architecture = self.build_architecture()

    @classmethod
    def from_checkpoint(cls, path: str, device: str = None):
        with open(f"{path}/init_arguments.json", "r") as f:
            init_arguments = json.load(f)
        if device is not None:
            init_arguments["device"] = device
        else:
            device = init_arguments["device"]
        model = cls(**init_arguments)
        model.load_state_dict(torch.load(f"{path}/model.pt"))
        return model.to(device)

    def get_n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, inputs, hidden_states=None, training: bool = True):
        raise NotImplementedError

    @torch.no_grad()
    def design_molecules(
        self,
        n_batches: int,
        batch_size: int,
        temperature: float,
        token2label: Dict[str, int],
    ):
        raise NotImplementedError
