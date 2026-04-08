"""
export_onnx.py
Export trained TinyBERT to ONNX format for hardware deployment.
Creates both FP32 and INT8 quantized versions.

Usage:
    python scripts/export_onnx.py --task sst2 --model_dir results/sst2
"""

import os
import sys
import argparse
import torch
from transformers import BertForSequenceClassification, BertTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def export_to_onnx(model_dir: str, task: str, output_dir: str = None):
    if output_dir is None:
        output_dir = model_dir

    device    = torch.device('cpu')   # export on CPU for compatibility
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model     = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval().to(device)

    # Dummy input for tracing
    dummy = tokenizer(
        "This is a test sentence for export",
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    onnx_path = os.path.join(output_dir, f"tinybert_{task}_fp32.onnx")

    print(f"Exporting FP32 ONNX model to: {onnx_path}")
    torch.onnx.export(
        model,
        (dummy['input_ids'], dummy['attention_mask'], dummy['token_type_ids']),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
            "token_type_ids": {0: "batch_size", 1: "seq_length"},
            "logits":         {0: "batch_size"}
        },
        opset_version=12,
        do_constant_folding=True
    )
    print(f"  FP32 model size: {os.path.getsize(onnx_path) / 1024**2:.1f} MB")

    # INT8 quantization for Raspberry Pi
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = os.path.join(output_dir, f"tinybert_{task}_int8.onnx")
        print(f"\nQuantizing to INT8: {int8_path}")
        quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)
        print(f"  INT8 model size: {os.path.getsize(int8_path) / 1024**2:.1f} MB")
        reduction = (1 - os.path.getsize(int8_path) / os.path.getsize(onnx_path)) * 100
        print(f"  Size reduction: {reduction:.0f}%")
    except ImportError:
        print("  onnxruntime not installed — skipping INT8 quantization")
        print("  Run: pip install onnxruntime")

    print(f"\nExport complete. Files saved to: {output_dir}")
    print("Next steps:")
    print("  Raspberry Pi: copy tinybert_{task}_int8.onnx → pip install onnxruntime")
    print("  Arduino:      run scripts/export_arduino.py to generate C arrays")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       type=str, required=True)
    parser.add_argument("--model_dir",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    export_to_onnx(args.model_dir, args.task, args.output_dir)
