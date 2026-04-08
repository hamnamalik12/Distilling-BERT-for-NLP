# TinyBERT: Distilling BERT for Natural Language Understanding
## Replication + Edge Hardware Deployment

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

> Replication of the paper: *TinyBERT: Distilling BERT for Natural Language Understanding* (Jiao et al., EMNLP 2020)  
> Extended with hardware deployment on **Raspberry Pi 4** and **Arduino Uno** (Phase 2).

---

## Project Structure

```
tinybert_project/
│
├── src/
│   ├── config.py            # All hyperparameters in one place
│   ├── dataset.py           # SST-2 dataset loader
│   ├── model.py             # TinyBERT student model definition
│   ├── losses.py            # All 4 distillation loss functions (Eq. 7-10)
│   ├── train.py             # Full training loop
│   └── evaluate.py          # Evaluation and benchmarking
│
├── scripts/
│   ├── setup_env.bat        # One-click Anaconda environment setup (Windows)
│   ├── train_all.bat        # Train on all GLUE tasks
│   └── export_onnx.py       # Export model for hardware deployment
│
│
├── results/                 # Auto-generated training results
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Setup Environment (Anaconda)
```bash
# Run this once
scripts\setup_env.bat
```

### 2. Train TinyBERT on SST-2
```bash
conda activate tinybert
python src/train.py --task sst2
```

### 3. Evaluate and Compare with Paper
```bash
python src/evaluate.py --task sst2 --model_dir results/sst2
```

---

## Results

| Model | Params | Speedup | SST-2 | CoLA | Avg GLUE |
|-------|--------|---------|-------|------|----------|
| BERT-BASE (Teacher) | 109M | 1.0x | 93.4 | 52.8 | 79.5 |
| TinyBERT4 (Paper)   | 14.5M | 9.4x | 92.6 | 44.1 | 77.0 |
| TinyBERT4 (Ours)    | 14.5M | 9.4x | 81.7  | -  | -  |

*Results updated after training completes.*

---

## Hardware Deployment (Phase 2)

- **Raspberry Pi 4** — Full ONNX INT8 quantized inference
- **Arduino Uno** — Split inference (embedding lookup + final classification layer)

See `report/` for detailed hardware implementation plan.

---

## Paper Reference

```bibtex
@inproceedings{jiao2020tinybert,
  title={TinyBERT: Distilling BERT for Natural Language Understanding},
  author={Jiao, Xiaoqi and Yin, Yichun and Shang, Lifeng and Jiang, Xin and
          Chen, Xiao and Li, Linlin and Wang, Fang and Liu, Qun},
  booktitle={Findings of EMNLP},
  year={2020}
}
```
