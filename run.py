"""
CLI runner: single image prediction or batch random evaluation.

Usage:
    python run.py --image path/to/image.jpg
    python run.py --random [--n 50]
"""

import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model import SimpleUNet
from dataset import TuSimpleDataset
from metrics import compute_all_metrics, print_metrics

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
IMG_DIR = "data/tusimple/processed/images"
MASK_DIR = "data/tusimple/processed/masks"
MODEL_PATH = "checkpoints/model_best.pth"
OUTPUT_DIR = "outputs"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_W, MODEL_H = 256, 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
model = SimpleUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# ──────────────────────────────────────────────
# PREDICT SINGLE IMAGE  (returns pred at orig size)
# ──────────────────────────────────────────────
def predict_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")

    h, w = img.shape[:2]

    inp = cv2.resize(img, (MODEL_W, MODEL_H)).astype(np.float32) / 255.0
    inp_t = torch.tensor(inp).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_small = model(inp_t)[0][0].cpu().numpy()   # (128, 256)

    pred_full = cv2.resize(pred_small, (w, h))
    mask = (pred_full > 0.5).astype(np.uint8)

    # Overlay lanes in red on original image
    overlay = img.copy()
    red_layer = np.zeros_like(img)
    red_layer[mask == 1] = (0, 0, 255)
    result = cv2.addWeighted(img, 0.7, red_layer, 0.5, 0)

    save_path = os.path.join(OUTPUT_DIR, "result.png")
    cv2.imwrite(save_path, result)
    print(f"Result saved → {save_path}")

    return pred_small, mask, img


# ──────────────────────────────────────────────
# BATCH RANDOM EVALUATION
# ──────────────────────────────────────────────
def run_random(n: int = 50):
    dataset = TuSimpleDataset(IMG_DIR, MASK_DIR, augment=False)
    n = min(n, len(dataset))

    keys = list(compute_all_metrics(np.zeros((1,1)), np.zeros((1,1))).keys())
    history = {k: [] for k in keys}

    for i in range(n):
        img_t, gt_t = dataset[i]

        inp = img_t.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(inp)[0][0].cpu().numpy()

        gt = gt_t[0].numpy()
        m = compute_all_metrics(pred, gt)
        for k, v in m.items():
            history[k].append(v)

    means = {k: float(np.mean(v)) for k, v in history.items()}
    print_metrics(means, prefix=f"Batch ({n} samples) Mean ")

    # ── Plot all metrics per sample ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (k, vals) in zip(axes, history.items()):
        ax.plot(vals, linewidth=0.8, alpha=0.8)
        ax.axhline(np.mean(vals), color="red", linestyle="--", linewidth=1,
                   label=f"Mean {np.mean(vals):.3f}")
        ax.set_title(k.replace("_", " ").title())
        ax.set_xlabel("Sample")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide empty subplot (we have 5 metrics, 6 subplots)
    axes[-1].axis("off")

    plt.suptitle(f"Model Evaluation  (n={n})", fontsize=13)
    plt.tight_layout()
    graph_path = os.path.join(OUTPUT_DIR, "metrics_all.png")
    plt.savefig(graph_path, dpi=150)
    plt.close()
    print(f"Metrics graph saved → {graph_path}")

    # ── Bar chart of means ──
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    bars = ax2.bar(list(means.keys()), list(means.values()), color=colors)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Mean Metrics Summary")
    for bar, val in zip(bars, means.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "metrics_summary.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Summary bar chart → {bar_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane detection runner")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--random", action="store_true", help="Batch eval on random samples")
    parser.add_argument("--n", type=int, default=50, help="Number of samples for --random")
    args = parser.parse_args()

    if args.image:
        pred_small, mask, img = predict_image(args.image)
        # Metrics against self (no GT here) — just show raw confidence
        conf = float(np.mean(pred_small))
        lane_pct = float(mask.mean() * 100)
        print(f"\nMean prediction confidence : {conf:.4f}")
        print(f"Lane pixel coverage        : {lane_pct:.2f} %")
        print("  (Pass --random to get full metrics vs ground truth)")

    elif args.random:
        run_random(n=args.n)

    else:
        print("Usage:")
        print("  python run.py --image path/to/image.jpg")
        print("  python run.py --random [--n 50]")