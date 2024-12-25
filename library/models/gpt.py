"""
Mostly borrowed from: https://github.com/karpathy/nanoGPT
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from library import sampling
from library.data import data_utils
from library.models.clm import ChemicalLanguageModel
from library.models.gpt_modules.gpt_block import GPTBlock


class GPT(ChemicalLanguageModel):
    def __init__(
        self,
        n_heads: int,
        **shared_clm_args,
    ):
        self.n_heads = n_heads
        super().__init__(**shared_clm_args)

    def build_architecture(self) -> nn.ModuleDict:
        return nn.ModuleDict(
            dict(
                token_embedding=nn.Embedding(
                    self.vocab_size, self.model_dim, padding_idx=0
                ),
                pos_embedding=nn.Embedding(
                    self.sequence_length, self.model_dim, padding_idx=0
                ),
                dropout=nn.Dropout(self.dropout),
                transformer=nn.ModuleList(
                    [
                        GPTBlock(self.model_dim, self.n_heads, self.dropout)
                        for _ in range(self.n_layers)
                    ]
                ),
                layer_norm=nn.LayerNorm(self.model_dim),
                lm_head=nn.Linear(self.model_dim, self.vocab_size),
            )
        )

    def forward(
        self, inputs: torch.Tensor, hidden_states=None, training: bool = True
    ) -> torch.Tensor:
        pos = torch.arange(
            0, self.sequence_length, dtype=torch.int, device=self.device
        )  # (sequence_length)

        token_embeddings = self.architecture.token_embedding(
            inputs
        )  #  (batch_size, sequence_length, model_dim)
        pos_embeddings = self.architecture.pos_embedding(
            pos
        )  # (sequence_length, model_dim)
        x = self.architecture.dropout(
            token_embeddings + pos_embeddings
        )  # (batch_size, sequence_length, model_dim)
        for block in self.architecture.transformer:
            x = block(x)  # (batch_size, sequence_length, model_dim)
        x = self.architecture.layer_norm(x)

        logits = self.architecture.lm_head(
            x
        )  # (batch_size, sequence_length, vocab_size)
        return logits

    @torch.no_grad()
    def design_molecules(
        self,
        n_batches: int,
        batch_size: int,
        temperature: float,
        token2label: Dict[str, int],
        top_k: int = 33,
        top_p: float = 1,
    ) -> Tuple[List[str], np.array]:
        self.to(self.device, dtype=torch.float32)
        self.eval()
        label2token = {v: k for k, v in token2label.items()}
        designs, loglikelihoods = list(), list()
        for _ in range(n_batches):
            design = torch.zeros(
                batch_size,
                self.sequence_length,
                dtype=torch.int32,
                device=self.device,
            )
            design[:, 0] = token2label[data_utils.beg_token]
            batch_loglikelihoods = list()
            for token_idx in range(self.sequence_length - 1):
                preds = self.forward(
                    design, None, training=False
                )  # (batch_size, sequence_length, vocab_size)
                preds = preds[:, token_idx, :].squeeze(1)  # (batch_size, vocab_size)
                if top_k != 33:
                    preds = sampling.top_k_filtering(preds, top_k)
                elif top_p != 1:
                    preds = sampling.top_p_filtering(preds, top_p)

                # preds = preds / temperature
                # probas = F.softmax(preds, -1)
                # next_token = torch.multinomial(probas, num_samples=1).squeeze(
                #     1
                # )  # (batch_size, 1)
                next_token, loglikelihood = sampling.temperature_sampling(
                    preds, temperature
                )
                design[:, token_idx + 1] = next_token

                batch_loglikelihoods.append(loglikelihood)

            designs.append(design)
            batch_loglikelihoods = torch.vstack(batch_loglikelihoods).T
            loglikelihoods.append(batch_loglikelihoods)

        designs = torch.cat(designs, 0)[:, 1:].cpu().numpy().tolist()
        end_index = token2label[data_utils.end_token]
        designs = [
            design[: design.index(end_index)] if end_index in design else ""
            for design in designs
        ]
        loglikelihoods = torch.cat(loglikelihoods, 0).detach().cpu().numpy().tolist()
        loglikelihoods = [
            loglikelihoods[: len(design) + 1]
            for loglikelihoods, design in zip(loglikelihoods, designs)
        ]
        mean_loglikelihoods = [np.mean(ll) for ll in loglikelihoods]
        designs = [
            "".join([label2token[label] for label in design]) for design in designs
        ]
        return designs, mean_loglikelihoods
