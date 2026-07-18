"""
Developer B work package: Inference / sampling.

Implements autoregressive generation from a seed primer sequence using a
MusicSequenceModel, with temperature scaling, top-k and/or nucleus (top-p)
sampling, and a repetition penalty to reduce the likelihood of repetitive loops
that are a common failure mode in symbolic music generation (see README.md,
Point 5 of the Kaggle analysis framework).

This module depends only on the model's forward() interface (input_ids,
padding_mask -> logits) and shared/config.py + shared/vocabulary.py. It does not
import developer_a.dataset or developer_a.train.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F

from shared.config import DATA_CONFIG


def apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: List[int], penalty: float
) -> torch.Tensor:
    """
    Penalizes tokens that have already appeared in the generated sequence by
    dividing their logit (if positive) or multiplying it (if negative), following
    the approach used in CTRL / GPT-style repetition penalties. A penalty of 1.0
    is a no-op; values around 1.1-1.3 are typically effective for symbolic music.
    """
    if penalty == 1.0 or not generated_ids:
        return logits

    unique_ids = set(generated_ids)
    for token_id in unique_ids:
        current_logit = logits[token_id]
        if current_logit > 0:
            logits[token_id] = current_logit / penalty
        else:
            logits[token_id] = current_logit * penalty
    return logits


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("inf"),
) -> torch.Tensor:
    """
    Filters a 1D logits tensor using top-k and/or nucleus (top-p) filtering.
    Adapted from the standard Hugging Face implementation pattern. Set top_k=0
    and top_p=0.0 to disable each filter independently.

    Args:
        logits: 1D tensor of shape (vocab_size,).
        top_k: keep only the top_k highest-probability tokens (0 = disabled).
        top_p: keep the smallest set of tokens whose cumulative probability
            exceeds top_p (0.0 = disabled).
    """
    assert logits.dim() == 1

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k)[0][..., -1]
        logits = torch.where(logits < threshold, torch.full_like(logits, filter_value), logits)

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Identify tokens to remove: those beyond the cumulative probability
        # threshold. Shift right to always keep at least one token.
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    primer_ids: List[int],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    max_context_length: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> List[int]:
    """
    Generates a new token sequence autoregressively from a seed primer.

    Args:
        model: a MusicSequenceModel in eval mode.
        primer_ids: seed token ids to condition generation on (e.g. BOS plus a
            short opening phrase extracted from a reference piece).
        max_new_tokens: number of tokens to generate beyond the primer.
        temperature: softmax temperature. Lower than 1.0 sharpens the
            distribution (more conservative/repetitive); higher than 1.0 flattens
            it (more diverse/riskier). 0.8-1.1 is a reasonable starting range.
        top_k: top-k filtering parameter (0 disables).
        top_p: nucleus sampling parameter (0.0 disables).
        repetition_penalty: penalty applied to previously generated tokens
            (1.0 disables). This is the primary lever for avoiding repetitive
            loops, in combination with top_p.
        max_context_length: truncate the context window fed to the model to this
            length (sliding window) if generation exceeds the model's trained
            max_sequence_length. Defaults to shared.config.DATA_CONFIG.max_sequence_length.

    Returns:
        The full list of token ids: primer_ids followed by newly generated ids.
    """
    device = device or next(model.parameters()).device
    max_context_length = max_context_length or DATA_CONFIG.max_sequence_length

    assert temperature > 0.0, "temperature must be strictly positive"

    generated = list(primer_ids)
    eos_token_id = DATA_CONFIG.eos_token_id

    for _ in range(max_new_tokens):
        context = generated[-max_context_length:]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)

        logits = model(input_ids)  # (1, seq_len, vocab_size)
        next_token_logits = logits[0, -1, :].clone()  # (vocab_size,)

        next_token_logits = next_token_logits / temperature
        next_token_logits = apply_repetition_penalty(next_token_logits, generated, repetition_penalty)
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

        probabilities = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, num_samples=1).item()

        generated.append(next_token_id)

        if next_token_id == eos_token_id:
            break

    return generated
