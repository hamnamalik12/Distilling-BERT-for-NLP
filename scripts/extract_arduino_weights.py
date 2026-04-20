"""
extract_arduino_weights.py
Extracts TinyBERT classifier weights and saves as C header for Arduino.
"""

import torch
import numpy as np
import argparse
import os
from transformers import BertForSequenceClassification

def extract_weights(model_dir):
    print(f"Loading model from: {model_dir}")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Get classifier weights shape: (num_labels, hidden_size) = (2, 312)
    weights = model.classifier.weight.detach().numpy()
    bias    = model.classifier.bias.detach().numpy()

    print(f"Classifier weight shape : {weights.shape}")
    print(f"Classifier bias shape   : {bias.shape}")
    print(f"Hidden size             : {weights.shape[1]}")
    print(f"Num labels              : {weights.shape[0]}")

    num_labels  = weights.shape[0]
    hidden_size = weights.shape[1]

    # Calculate memory usage
    weight_bytes = num_labels * hidden_size * 4  # float32
    bias_bytes   = num_labels * 4
    total_kb     = (weight_bytes + bias_bytes) / 1024
    print(f"\nMemory needed on Arduino : {total_kb:.1f} KB")
    print(f"Arduino Flash available  : 32 KB")

    if total_kb > 28:
        print("WARNING: May not fit on Arduino Uno!")
    else:
        print("Fits comfortably on Arduino Uno.")

    # Write C header file
    out_path = os.path.join(model_dir, "tinybert_weights.h")

    with open(out_path, "w") as f:
        f.write("// Auto-generated TinyBERT classifier weights\n")
        f.write("// For Arduino Uno split inference\n")
        f.write("// Generated from: " + model_dir + "\n\n")
        f.write("#ifndef TINYBERT_WEIGHTS_H\n")
        f.write("#define TINYBERT_WEIGHTS_H\n\n")
        f.write(f"#define HIDDEN_SIZE {hidden_size}\n")
        f.write(f"#define NUM_LABELS  {num_labels}\n\n")

        # Classifier weights array
        f.write("const float classifier_weights[NUM_LABELS][HIDDEN_SIZE] = {\n")
        for i in range(num_labels):
            vals = ", ".join([f"{v:.6f}f" for v in weights[i]])
            label = "NEGATIVE" if i == 0 else "POSITIVE"
            f.write(f"  // Label {i}: {label}\n")
            f.write(f"  {{{vals}}},\n")
        f.write("};\n\n")

        # Classifier bias array
        f.write("const float classifier_bias[NUM_LABELS] = {\n")
        for i in range(num_labels):
            label = "NEGATIVE" if i == 0 else "POSITIVE"
            f.write(f"  {bias[i]:.6f}f,  // Label {i}: {label}\n")
        f.write("};\n\n")

        f.write("#endif // TINYBERT_WEIGHTS_H\n")

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nSaved to  : {out_path}")
    print(f"File size : {size_kb:.1f} KB")
    print("\nNext step : copy tinybert_weights.h into your Arduino sketch folder")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True,
                        help="Path to trained model directory")
    args = parser.parse_args()
    extract_weights(args.model_dir)