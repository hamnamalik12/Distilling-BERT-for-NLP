"""
config.py
All hyperparameters and settings for TinyBERT replication.
Based on Section 4.2 of the paper.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TinyBERTConfig:
    # ── Student model architecture (Table 1, Section 4.2) ──────────────
    vocab_size:            int = 30522
    hidden_size:           int = 312       # d' in paper
    num_hidden_layers:     int = 4         # M in paper
    num_attention_heads:   int = 12        # h in paper
    intermediate_size:     int = 1200      # d'_i in paper
    max_position_embeddings: int = 512
    hidden_dropout_prob:   float = 0.1
    attention_probs_dropout_prob: float = 0.1
    num_labels:            int = 2

    # ── Teacher model ────────────────────────────────────────────────────
    teacher_model: str = "textattack/bert-base-uncased-SST-2"
    teacher_hidden_size: int = 768         # d in paper
    teacher_num_layers:  int = 12          # N in paper

    # ── Layer mapping: student layer m -> teacher layer g(m) = 3*m ──────
    # Section 4.2: uniform strategy
    layer_map: Dict[int, int] = field(default_factory=lambda: {
        1: 3,
        2: 6,
        3: 9,
        4: 12
    })

    # ── Training hyperparameters (Section 4.2) ───────────────────────────
    batch_size:         int   = 32
    max_seq_length:     int   = 64         # 64 for single-sentence tasks
    num_epochs:         int   = 20
    learning_rate:      float = 5e-5
    warmup_steps:       int   = 100
    weight_decay:       float = 0.01
    max_grad_norm:      float = 1.0
    temperature:        float = 1.0        # t in Equation 10

    # ── Loss weights ─────────────────────────────────────────────────────
    alpha: float = 0.7                     # weight for prediction distillation loss
    # Final loss = embd + attn + hidn + alpha*pred + (1-alpha)*task

    # ── Data augmentation (Algorithm 1, Section 3.2) ─────────────────────
    aug_pt:  float = 0.4                   # replacement threshold probability
    aug_na:  int   = 20                    # augmented samples per example
    aug_k:   int   = 15                    # candidate set size

    # ── Paths ────────────────────────────────────────────────────────────
    save_dir:    str = "results"
    log_dir:     str = "logs"
    data_dir:    str = "glue_data"

    # ── Hardware ─────────────────────────────────────────────────────────
    num_workers:  int = 2
    pin_memory:   bool = True
    seed:         int = 42


# Task-specific settings
TASK_CONFIGS = {
    "sst2": {
        "huggingface_name": "sst2",
        "teacher_model":    "textattack/bert-base-uncased-SST-2",
        "max_seq_length":   64,
        "num_labels":       2,
        "metric":           "accuracy",
        "text_col":         "sentence",
        "label_col":        "label",
        "val_split":        "validation",
    },
    "mrpc": {
        "huggingface_name": "mrpc",
        "teacher_model":    "textattack/bert-base-uncased-MRPC",
        "max_seq_length":   128,
        "num_labels":       2,
        "metric":           "f1",
        "text_col":         ("sentence1", "sentence2"),
        "label_col":        "label",
        "val_split":        "validation",
    },
    "cola": {
        "huggingface_name": "cola",
        "teacher_model":    "textattack/bert-base-uncased-CoLA",
        "max_seq_length":   64,
        "num_labels":       2,
        "metric":           "matthews_correlation",
        "text_col":         "sentence",
        "label_col":        "label",
        "val_split":        "validation",
    },
}

# Paper results for comparison (Table 1)
PAPER_RESULTS = {
    "bert_base": {"sst2": 93.4, "cola": 52.8, "mrpc": 87.5, "avg": 79.5},
    "tinybert4": {"sst2": 92.6, "cola": 44.1, "mrpc": 86.4, "avg": 77.0},
}
