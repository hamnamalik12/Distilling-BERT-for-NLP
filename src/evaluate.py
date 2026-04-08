"""
evaluate.py
Model evaluation, benchmarking, and results comparison.

Usage:
    python src/evaluate.py --task sst2 --model_dir results/sst2
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TinyBERTConfig, TASK_CONFIGS, PAPER_RESULTS
from dataset import get_dataloaders


def evaluate_model(model, val_loader, device, task: str) -> float:
    """
    Run evaluation and return the task metric score.
    SST-2/MRPC → accuracy, CoLA → Matthews correlation.
    """
    task_cfg = TASK_CONFIGS[task]
    metric   = task_cfg["metric"]

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            tids = batch['token_type_ids'].to(device)
            lbls = batch['label']

            logits = model(input_ids=ids,
                           attention_mask=mask,
                           token_type_ids=tids).logits
            preds  = logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(lbls.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    if metric == "accuracy":
        return accuracy_score(all_labels, all_preds)
    elif metric == "matthews_correlation":
        return matthews_corrcoef(all_labels, all_preds)
    elif metric == "f1":
        return f1_score(all_labels, all_preds)
    else:
        return accuracy_score(all_labels, all_preds)


def benchmark_speed(model, tokenizer, device,
                    max_length: int = 64,
                    n_runs: int = 100) -> dict:
    """
    Measure inference latency and throughput.
    Matches paper's benchmarking approach (single sample).
    """
    model.eval()
    dummy = tokenizer(
        "The movie was absolutely fantastic and I loved every moment",
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    ids  = dummy['input_ids'].to(device)
    mask = dummy['attention_mask'].to(device)
    tids = dummy['token_type_ids'].to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids=ids, attention_mask=mask, token_type_ids=tids)

    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(input_ids=ids, attention_mask=mask, token_type_ids=tids)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "mean_latency_ms":   round(np.mean(times), 2),
        "std_latency_ms":    round(np.std(times), 2),
        "min_latency_ms":    round(np.min(times), 2),
        "max_latency_ms":    round(np.max(times), 2),
        "throughput_per_s":  round(1000 / np.mean(times), 1),
        "n_runs":            n_runs
    }


def count_parameters(model) -> dict:
    """Count total and trainable parameters."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params":       total,
        "trainable_params":   trainable,
        "total_params_M":     round(total / 1e6, 1),
        "model_size_mb":      round(total * 4 / 1024**2, 1)  # float32
    }


def compare_with_paper(our_results: dict) -> str:
    """Generate comparison table vs paper results."""
    lines = [
        "\n" + "=" * 70,
        "  RESULTS COMPARISON WITH PAPER (Table 1)",
        "=" * 70,
        f"  {'Model':<25} {'SST-2':>8} {'CoLA':>8} {'MRPC':>8} {'Params':>10}",
        "-" * 70,
    ]

    paper_bert = PAPER_RESULTS["bert_base"]
    paper_tiny = PAPER_RESULTS["tinybert4"]

    lines.append(
        f"  {'BERT-BASE (Teacher)':<25} "
        f"{paper_bert.get('sst2','—'):>8} "
        f"{paper_bert.get('cola','—'):>8} "
        f"{paper_bert.get('mrpc','—'):>8} "
        f"{'109M':>10}"
    )
    lines.append(
        f"  {'TinyBERT4 (Paper)':<25} "
        f"{paper_tiny.get('sst2','—'):>8} "
        f"{paper_tiny.get('cola','—'):>8} "
        f"{paper_tiny.get('mrpc','—'):>8} "
        f"{'14.5M':>10}"
    )

    ours_sst2 = our_results.get("sst2", {}).get("val_score", "—")
    ours_cola = our_results.get("cola", {}).get("val_score", "—")
    ours_mrpc = our_results.get("mrpc", {}).get("val_score", "—")

    def fmt(v):
        return f"{v*100:.1f}" if isinstance(v, float) else str(v)

    lines.append(
        f"  {'TinyBERT4 (Ours)':<25} "
        f"{fmt(ours_sst2):>8} "
        f"{fmt(ours_cola):>8} "
        f"{fmt(ours_mrpc):>8} "
        f"{'14.5M':>10}"
    )
    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",      type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    args = parser.parse_args()

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model     = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)
    cfg       = TinyBERTConfig()

    _, val_loader = get_dataloaders(args.task, tokenizer, cfg)

    print("\nEvaluating...")
    score  = evaluate_model(model, val_loader, device, args.task)
    speed  = benchmark_speed(model, tokenizer, device)
    params = count_parameters(model)

    print(f"\n  Task:          {args.task.upper()}")
    print(f"  Val score:     {score:.4f}")
    print(f"  Parameters:    {params['total_params_M']}M")
    print(f"  Latency:       {speed['mean_latency_ms']} ms ± {speed['std_latency_ms']} ms")
    print(f"  Throughput:    {speed['throughput_per_s']} samples/s")

    results = {args.task: {"val_score": score, "speed": speed, "params": params}}
    print(compare_with_paper(results))
