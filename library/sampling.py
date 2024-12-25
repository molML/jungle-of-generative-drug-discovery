import torch
import torch.nn.functional as F


@torch.inference_mode()
def temperature_sampling(preds, temperature):
    t_preds = preds / temperature
    t_probas = F.softmax(t_preds, -1)
    next_token = torch.multinomial(t_probas, num_samples=1).squeeze(1).int()

    probas = F.softmax(preds, -1)
    log_probas = torch.log(probas)
    loglikelihood = log_probas[torch.arange(log_probas.size(0)), next_token]
    return next_token, loglikelihood


@torch.inference_mode()
def top_k_filtering(preds, top_k):
    top_k_preds = preds.topk(k=top_k, dim=-1)
    new_dist = torch.zeros_like(preds) + float("-inf")
    return torch.scatter(
        new_dist, dim=-1, index=top_k_preds.indices, src=top_k_preds.values
    )


@torch.inference_mode()
def top_p_filtering(preds, top_p):
    sorted_preds, _ = torch.sort(preds, dim=-1, descending=True)

    probas = F.softmax(preds, -1)
    sorted_probas, sorted_indices = torch.sort(probas, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probas, dim=-1)
    sorted_indices_to_remove = cumulative_probs >= top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    sorted_preds[sorted_indices_to_remove] = float("-inf")
    return torch.gather(sorted_preds, 1, sorted_indices.argsort(-1))
