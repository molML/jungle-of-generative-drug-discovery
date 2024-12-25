from typing import Dict, List, Tuple

import numpy as np
import torch

from library import sampling
from library.data import data_utils
from library.models.clm import ChemicalLanguageModel


class AutoRegressiveChemicalLanguageModel(ChemicalLanguageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_hidden_states(self, batch_size: int):
        raise NotImplementedError

    # @torch.no_grad()
    @torch.inference_mode()
    def design_molecules(
        self,
        n_batches: int,
        batch_size: int,
        temperature: float,
        token2label: Dict[str, int],
        top_k: int = 33,
        top_p: float = 1,
    ) -> Tuple[List[str], List[float]]:
        if "s4" in self.architecture.keys():
            # this should run only for s4
            for module in self.architecture.s4.modules():
                if hasattr(module, "setup_step"):
                    module.setup_step()

        self.to(self.device, dtype=torch.float32)
        self.eval()
        label2token = {v: k for k, v in token2label.items()}
        designs, loglikelihoods = list(), list()
        for _ in range(n_batches):
            hidden_state = self.initialize_hidden_states(batch_size)
            current_token = torch.zeros(
                batch_size, dtype=torch.int32, device=self.device
            )
            current_token = token2label[data_utils.beg_token] + current_token

            batch_designs, batch_loglikelihoods = list(), list()
            for __ in range(
                self.sequence_length - 1
            ):  # -1 since <BEG> is already added
                preds, hidden_state = self.forward(
                    current_token, hidden_state, training=False
                )  # (batch_size, 1, vocab_size)
                preds = preds.squeeze(1)  # (batch_size, vocab_size)
                if top_k != 33:
                    preds = sampling.top_k_filtering(preds, top_k)
                elif top_p != 1:
                    preds = sampling.top_p_filtering(preds, top_p)

                next_token, loglikelihood = sampling.temperature_sampling(
                    preds, temperature
                )
                batch_designs.append(next_token)
                batch_loglikelihoods.append(loglikelihood)
                current_token = next_token

            batch_designs = torch.vstack(batch_designs).T
            designs.append(batch_designs)

            batch_loglikelihoods = torch.vstack(batch_loglikelihoods).T
            loglikelihoods.append(batch_loglikelihoods)

        designs = torch.cat(designs, 0).cpu().numpy().tolist()
        end_index = token2label[data_utils.end_token]
        designs = [
            design[: design.index(end_index)] if end_index in design else ""
            for design in designs
        ]
        loglikelihoods = torch.cat(loglikelihoods, 0).detach().cpu().numpy().tolist()
        mean_loglikelihoods = [
            np.mean(loglikelihoods[: len(design) + 1]).astype(float)
            for loglikelihoods, design in zip(loglikelihoods, designs)
        ]
        designs = [
            "".join([label2token[label] for label in design]) for design in designs
        ]

        return designs, mean_loglikelihoods
