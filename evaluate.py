"""
Evaluate trained model on a subset of the dataset.

Usage:
    python evaluate.py [--n 200] [--model checkpoints/model_best.pth]

Outputs:
    outputs/eval_metrics_summary.png   ← bar chart of mean metrics
    outputs/eval_per_sample.png        ← per-sample IoU + Dice curves
    outputs/eval_results.csv           ← per-sample scores
"""

import os
import argparse
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from model import SimpleUNet
from metrics import compute_all_metrics, print_metrics

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
IMG_DIR = "data/tusimple/processed/images"
MASK_DIR = "data/tusimple/processed/masks"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=200, help="Number of samples to evaluate")
parser.add_argument("--model", type=str, default="checkpoints/model_best.pth")
args = parser.parse_args()

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
model = SimpleUNet().to(DEVICE)
model.load_state_dict(torch.load(args.model, map_location=DEVICE))
model.eval()
print(f"Model loaded: {args.model}")

# ──────────────────────────────────────────────
# GET PAIRED FILES
# ──────────────────────────────────────────────
img_files = set(os.listdir(IMG_DIR))
mask_files = set(os.listdir(MASK_DIR))
common = sorted(img_files & mask_files)[: args.n]
print(f"Evaluating on {len(common)} samples …\n")

# ──────────────────────────────────────────────
# EVALUATE
# ──────────────────────────────────────────────
all_metrics = {k: [] for k in ["iou", "dice", "pixel_accuracy", "precision", "recall"]}
csv_rows = []

for fname in common:
    img = cv2.imread(os.path.join(IMG_DIR, fname))
    mask = cv2.imread(os.path.join(MASK_DIR, fname), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        continue

    img_r = cv2.resize(img, (256, 128)).astype(np.float32) / 255.0
    mask_r = cv2.resize(mask, (256, 128)).astype(np.float32) / 255.0

    inp = torch.tensor(img_r).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(inp)[0][0].cpu().numpy()

    m = compute_all_metrics(pred, mask_r)
    for k, v in m.items():
        all_metrics[k].append(v)

    csv_rows.append({"file": fname, **{k: f"{v:.4f}" for k, v in m.items()}})

# ──────────────────────────────────────────────
# AGGREGATE
# ──────────────────────────────────────────────
means = {k: float(np.mean(v)) for k, v in all_metrics.items()}
print_metrics(means, prefix="Mean ")

# ──────────────────────────────────────────────
# SAVE CSV
# ──────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "eval_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "iou", "dice", "pixel_accuracy", "precision", "recall"])
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\nPer-sample CSV saved → {csv_path}")

# ──────────────────────────────────────────────
# PLOT 1: Mean metrics bar chart
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
labels = list(means.keys())
values = list(means.values())
bars = ax.bar(labels, values, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title(f"Evaluation Summary  (n={len(csv_rows)})")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eval_metrics_summary.png"), dpi=150)
plt.close()

# ──────────────────────────────────────────────
# PLOT 2: Per-sample IoU + Dice line chart
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(all_metrics["iou"], label="IoU", alpha=0.7, linewidth=0.8)
ax.plot(all_metrics["dice"], label="Dice", alpha=0.7, linewidth=0.8)
ax.axhline(means["iou"], color="blue", linestyle="--", linewidth=1, alpha=0.5, label=f"Mean IoU {means['iou']:.3f}")
ax.axhline(means["dice"], color="orange", linestyle="--", linewidth=1, alpha=0.5, label=f"Mean Dice {means['dice']:.3f}")
ax.set_xlabel("Sample index")
ax.set_ylabel("Score")
ax.set_title("Per-sample IoU and Dice")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eval_per_sample.png"), dpi=150)
plt.close()

print("Plots saved → outputs/eval_metrics_summary.png")
print("           → outputs/eval_per_sample.png")
print("\nEvaluation DONE ✔")