"""
model.py
TinyBERT student model definition.
Architecture from Section 4.2 of the paper:
  M=4 layers, d'=312, d'_i=1200, h=12 heads → ~14.5M parameters
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification
from config import TinyBERTConfig


def build_student(cfg: TinyBERTConfig) -> BertForSequenceClassification:
    """
    Build TinyBERT4 student model.
    14.5M parameters as described in Section 4.2.
    """
    bert_cfg = BertConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        max_position_embeddings=cfg.max_position_embeddings,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
        num_labels=cfg.num_labels
    )
    model = BertForSequenceClassification(bert_cfg)

    total = sum(p.numel() for p in model.parameters())
    print(f"  Student (TinyBERT4) parameters: {total:,}  (~14.5M expected)")
    return model


def build_teacher(task: str,
                  cfg: TinyBERTConfig,
                  device: torch.device) -> BertForSequenceClassification:
    """
    Load fine-tuned BERT-BASE teacher for a specific task.
    Teacher is frozen — no gradients computed through it.
    """
    from config import TASK_CONFIGS
    teacher_name = TASK_CONFIGS[task]["teacher_model"]

    print(f"  Loading teacher: {teacher_name}")
    teacher = BertForSequenceClassification.from_pretrained(teacher_name)
    teacher = teacher.to(device)
    teacher.eval()

    # Freeze all teacher parameters
    for p in teacher.parameters():
        p.requires_grad = False

    total = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher (BERT-BASE) parameters: {total:,}  (~109M expected)")
    return teacher


class HiddenProjection(nn.Module):
    """
    Linear projection W_h from student hidden size to teacher hidden size.
    Used in Equations 8 and 9 of the paper.
    Student: d'=312  →  Teacher: d=768
    """
    def __init__(self, student_size: int = 312, teacher_size: int = 768):
        super().__init__()
        self.proj = nn.Linear(student_size, teacher_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
