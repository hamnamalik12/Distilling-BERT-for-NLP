import matplotlib.pyplot as plt
import numpy as np

# Accuracy Comparison
models = ['BERT-BASE', 'TinyBERT4 (Paper)', 'TinyBERT4 (Ours)']
sst2_scores = [93.4, 92.6, 81.7]

plt.figure(figsize=(8,5))
plt.bar(models, sst2_scores, color=['blue', 'green', 'orange'])
plt.ylabel('Validation Accuracy (%)')
plt.title('SST-2 Accuracy Comparison')
plt.ylim(0, 100)
for i, v in enumerate(sst2_scores):
    plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
plt.show()

# Parameter Comparison
params = [109, 14.5, 14.5]  # in millions
plt.figure(figsize=(8,5))
plt.bar(models, params, color=['blue', 'green', 'orange'])
plt.ylabel('Parameters (Millions)')
plt.title('Model Parameter Comparison')
for i, v in enumerate(params):
    plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
plt.show()

# Latency & Throughput
metrics = ['Latency (ms)', 'Throughput (samples/s)']
values = [5.14, 194.4]
plt.figure(figsize=(6,4))
plt.bar(metrics, values, color=['red', 'purple'])
plt.title('TinyBERT Evaluation Metrics')
for i, v in enumerate(values):
    plt.text(i, v + (2 if i==0 else 5), str(round(v,2)), ha='center', fontweight='bold')
plt.show()