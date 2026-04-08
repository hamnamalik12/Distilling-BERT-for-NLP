"""
dataset.py
Dataset loading and preprocessing for GLUE tasks.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
from config import TinyBERTConfig, TASK_CONFIGS


class GLUEDataset(Dataset):
    """
    Generic GLUE task dataset.
    Handles both single-sentence and sentence-pair tasks.
    """

    def __init__(self, task: str, split: str, tokenizer: BertTokenizer,
                 max_length: int = 64):
        self.task_cfg   = TASK_CONFIGS[task]
        self.tokenizer  = tokenizer
        self.max_length = max_length

        raw = load_dataset("glue", self.task_cfg["huggingface_name"])
        self.data = raw[split]
        print(f"  Loaded {split} split: {len(self.data):,} examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item      = self.data[idx]
        text_col  = self.task_cfg["text_col"]
        label_col = self.task_cfg["label_col"]

        # Single-sentence vs sentence-pair tasks
        if isinstance(text_col, tuple):
            text_a = item[text_col[0]]
            text_b = item[text_col[1]]
            enc = self.tokenizer(
                text_a, text_b,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            enc = self.tokenizer(
                item[text_col],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc['token_type_ids'].squeeze(0),
            'label':          torch.tensor(item[label_col], dtype=torch.long)
        }


def get_dataloaders(task: str, tokenizer: BertTokenizer,
                    cfg: TinyBERTConfig):
    """Returns train and validation DataLoaders for a GLUE task."""

    task_cfg    = TASK_CONFIGS[task]
    max_length  = task_cfg["max_seq_length"]
    val_split   = task_cfg["val_split"]

    print(f"\nLoading {task.upper()} dataset...")
    train_ds = GLUEDataset(task, "train",    tokenizer, max_length)
    val_ds   = GLUEDataset(task, val_split,  tokenizer, max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )

    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val   batches: {len(val_loader):,}")
    return train_loader, val_loader
