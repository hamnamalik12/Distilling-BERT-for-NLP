"""
losses.py
All four distillation loss functions from the paper.

Equation 7  — Attention loss    (Lattn)
Equation 8  — Hidden state loss (Lhidn)
Equation 9  — Embedding loss    (Lembd)
Equation 10 — Prediction loss   (Lpred)
Equation 11 — Combined layer loss (Llayer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


def attention_loss(
    student_attns: Tuple[torch.Tensor],
    teacher_attns: Tuple[torch.Tensor],
    layer_map: Dict[int, int]
) -> torch.Tensor:
    """
    Equation 7: Attention-based distillation loss.

    Lattn = (1/h) * sum_i MSE(A_i^S, A_i^T)

    Uses UNNORMALIZED attention matrices (before softmax),
    as the paper states this gives faster convergence.

    Args:
        student_attns: tuple of (batch, heads, seq, seq) tensors, one per student layer
        teacher_attns: tuple of (batch, heads, seq, seq) tensors, one per teacher layer
        layer_map: {student_layer: teacher_layer} mapping

    Returns:
        Scalar attention loss averaged over mapped layers
    """
    loss = torch.tensor(0.0, device=student_attns[0].device)
    for s_idx, t_idx in layer_map.items():
        s_attn = student_attns[s_idx - 1]   # 0-indexed tuple
        t_attn = teacher_attns[t_idx - 1]
        loss += F.mse_loss(s_attn, t_attn)
    return loss / len(layer_map)


def hidden_loss(
    student_hiddens: Tuple[torch.Tensor],
    teacher_hiddens: Tuple[torch.Tensor],
    projection: nn.Module,
    layer_map: Dict[int, int]
) -> torch.Tensor:
    """
    Equation 8: Hidden states distillation loss.

    Lhidn = MSE(H^S * W_h, H^T)

    Projects student hidden states (312) to teacher space (768)
    using learnable linear transformation W_h.

    Args:
        student_hiddens: tuple of (batch, seq, 312) tensors
        teacher_hiddens: tuple of (batch, seq, 768) tensors
        projection: HiddenProjection module (312 → 768)
        layer_map: {student_layer: teacher_layer} mapping
    """
    loss = torch.tensor(0.0, device=student_hiddens[0].device)
    for s_idx, t_idx in layer_map.items():
        s_h = student_hiddens[s_idx]         # hidden_states[0] = embedding layer
        t_h = teacher_hiddens[t_idx]
        projected = projection(s_h)
        loss += F.mse_loss(projected, t_h)
    return loss / len(layer_map)


def embedding_loss(
    student_hiddens: Tuple[torch.Tensor],
    teacher_hiddens: Tuple[torch.Tensor],
    projection: nn.Module
) -> torch.Tensor:
    """
    Equation 9: Embedding layer distillation loss.

    Lembd = MSE(E^S * W_e, E^T)

    hidden_states[0] is the embedding layer output (before any transformer layers).

    Args:
        student_hiddens: tuple, index 0 = embedding output (batch, seq, 312)
        teacher_hiddens: tuple, index 0 = embedding output (batch, seq, 768)
        projection: HiddenProjection module (312 → 768)
    """
    s_emb = student_hiddens[0]
    t_emb = teacher_hiddens[0]
    return F.mse_loss(projection(s_emb), t_emb)


def prediction_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Equation 10: Prediction layer distillation loss.

    Lpred = CE(z^T / t, z^S / t)

    Soft cross-entropy between student and teacher logit distributions.
    Paper finds t=1 works well.

    Args:
        student_logits: (batch, num_labels)
        teacher_logits: (batch, num_labels)
        temperature:    scaling factor t (default=1.0 per paper)
    """
    s_prob = F.log_softmax(student_logits / temperature, dim=-1)
    t_prob = F.softmax(teacher_logits    / temperature, dim=-1)
    return F.kl_div(s_prob, t_prob, reduction='batchmean')


def combined_loss(
    student_out,
    teacher_out,
    labels: torch.Tensor,
    projection: nn.Module,
    layer_map: Dict[int, int],
    alpha: float = 0.7,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Equation 11: Combined distillation loss.

    Llayer = Lembd + Lattn + Lhidn + alpha*Lpred + (1-alpha)*Ltask

    Args:
        student_out: model output with attentions and hidden states
        teacher_out: teacher output with attentions and hidden states
        labels:      ground truth labels for task loss
        projection:  HiddenProjection module
        layer_map:   {student_layer: teacher_layer}
        alpha:       weight for prediction distillation (0.7 per paper)
        temperature: for prediction loss (1.0 per paper)

    Returns:
        total_loss: scalar tensor
        loss_components: dict with individual loss values for logging
    """
    l_embd = embedding_loss(
        student_out.hidden_states,
        teacher_out.hidden_states,
        projection
    )
    l_attn = attention_loss(
        student_out.attentions,
        teacher_out.attentions,
        layer_map
    )
    l_hidn = hidden_loss(
        student_out.hidden_states,
        teacher_out.hidden_states,
        projection,
        layer_map
    )
    l_pred = prediction_loss(
        student_out.logits,
        teacher_out.logits,
        temperature
    )
    l_task = F.cross_entropy(student_out.logits, labels)

    total = l_embd + l_attn + l_hidn + alpha * l_pred + (1 - alpha) * l_task

    components = {
        "embd": l_embd.item(),
        "attn": l_attn.item(),
        "hidn": l_hidn.item(),
        "pred": l_pred.item(),
        "task": l_task.item(),
        "total": total.item()
    }
    return total, components
