"""
train.py
Full TinyBERT training pipeline.

Usage:
    python src/train.py --task sst2
    python src/train.py --task mrpc
    python src/train.py --task cola
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# Add src to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TinyBERTConfig, TASK_CONFIGS
from dataset import get_dataloaders
from model import build_student, build_teacher, HiddenProjection
from losses import combined_loss
from evaluate import evaluate_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(task: str, cfg: TinyBERTConfig = None):
    if cfg is None:
        cfg = TinyBERTConfig()

    # Update config for this task
    task_cfg = TASK_CONFIGS[task]
    cfg.num_labels     = task_cfg["num_labels"]
    cfg.max_seq_length = task_cfg["max_seq_length"]

    # ── Setup ────────────────────────────────────────────────────────────
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = os.path.join(cfg.save_dir, task)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    print("=" * 60)
    print(f"  TinyBERT Training — Task: {task.upper()}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    print("=" * 60)

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(task, tokenizer, cfg)

    # ── Models ───────────────────────────────────────────────────────────
    print("\nBuilding models...")
    teacher    = build_teacher(task, cfg, device)
    student    = build_student(cfg).to(device)
    projection = HiddenProjection(cfg.hidden_size, cfg.teacher_hidden_size).to(device)

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = AdamW(
        list(student.parameters()) + list(projection.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        eps=1e-8
    )
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps
    )

    # ── Training log ─────────────────────────────────────────────────────
    history = {
        "task": task, "epochs": [], "best_val_score": 0.0,
        "config": {
            "batch_size": cfg.batch_size,
            "lr": cfg.learning_rate,
            "epochs": cfg.num_epochs,
            "alpha": cfg.alpha
        }
    }

    print(f"\nStarting training — {cfg.num_epochs} epochs\n")
    best_score = 0.0
    start_time = time.time()

    for epoch in range(cfg.num_epochs):
        student.train()
        projection.train()

        epoch_losses = {"embd": 0, "attn": 0, "hidn": 0,
                        "pred": 0, "task": 0, "total": 0}
        epoch_start = time.time()

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1:02d}/{cfg.num_epochs}",
                    ncols=90)

        for batch in pbar:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            tids = batch['token_type_ids'].to(device)
            lbls = batch['label'].to(device)

            # Teacher forward — no gradients needed
            with torch.no_grad():
                t_out = teacher(
                    input_ids=ids,
                    attention_mask=mask,
                    token_type_ids=tids,
                    output_attentions=True,
                    output_hidden_states=True
                )

            # Student forward
            s_out = student(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=tids,
                output_attentions=True,
                output_hidden_states=True
            )

            # Combined loss (Equation 11)
            loss, components = combined_loss(
                s_out, t_out, lbls, projection,
                cfg.layer_map, cfg.alpha, cfg.temperature
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] += components[k]

            pbar.set_postfix({"loss": f"{components['total']:.3f}"})

        # ── Per-epoch averages ────────────────────────────────────────────
        n = len(train_loader)
        avg = {k: v / n for k, v in epoch_losses.items()}
        epoch_time = time.time() - epoch_start

        # ── Validation ───────────────────────────────────────────────────
        val_score = evaluate_model(student, val_loader, device, task)

        print(f"  Epoch {epoch+1:02d} | "
              f"Loss: {avg['total']:.4f} | "
              f"Attn: {avg['attn']:.4f} | "
              f"Hidn: {avg['hidn']:.4f} | "
              f"Val {task_cfg['metric']}: {val_score:.4f} | "
              f"Time: {epoch_time:.0f}s")

        # ── Log ──────────────────────────────────────────────────────────
        history["epochs"].append({
            "epoch": epoch + 1,
            "losses": avg,
            "val_score": val_score,
            "time_s": epoch_time
        })

        # ── Save best ────────────────────────────────────────────────────
        if val_score > best_score:
            best_score = val_score
            history["best_val_score"] = best_score
            student.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  ✓ New best {task_cfg['metric']}: {best_score:.4f} — model saved")

    total_time = time.time() - start_time
    history["total_time_minutes"] = total_time / 60

    # ── Save training history ─────────────────────────────────────────────
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Best {task_cfg['metric']}: {best_score:.4f}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Model saved to: {save_dir}")
    print("=" * 60)

    return best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyBERT on GLUE task")
    parser.add_argument("--task",       type=str, default="sst2",
                        choices=["sst2", "mrpc", "cola"],
                        help="GLUE task to train on")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--save_dir",   type=str, default="results")
    args = parser.parse_args()

    cfg = TinyBERTConfig()
    cfg.save_dir = args.save_dir
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr

    train(args.task, cfg)
