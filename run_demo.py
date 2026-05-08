"""
Ultimate Lane Detection Demo

Menu:
    1. Image  — overlay + metrics + saved comparison
    2. Random — random from dataset with ground truth comparison
    3. Video  — process video file with confidence graph
    4. Live   — real-time webcam lane detection
    5. Evaluate All — full evaluation with all graphs and metrics
"""

import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime

from model import SimpleUNet
from metrics import compute_all_metrics, print_metrics

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
IMG_DIR    = "data/tusimple/processed/images"
MASK_DIR   = "data/tusimple/processed/masks"
MODEL_PATH = "checkpoints/model_best.pth"
OUTPUT_DIR = "outputs"
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
MODEL_W, MODEL_H = 256, 128

# ── Create all output subdirectories ──
os.makedirs(os.path.join(OUTPUT_DIR, "images"),  exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "graphs"),  exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "metrics"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "videos"),  exist_ok=True)

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
model = SimpleUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded  |  Device: {DEVICE}")


# ──────────────────────────────────────────────
# CORE FUNCTIONS
# ──────────────────────────────────────────────
def preprocess(img_bgr):
    img = cv2.resize(img_bgr, (MODEL_W, MODEL_H)).astype(np.float32) / 255.0
    return torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)


def get_lane_mask(prob_map, threshold=0.3):
    binary = (prob_map > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 30:
            clean[labels == i] = 255
    return clean


def predict(img_bgr):
    inp = preprocess(img_bgr)
    with torch.no_grad():
        prob = model(inp)[0][0].cpu().numpy()
    conf = float(np.mean(prob))
    lane_mask_small = get_lane_mask(prob)
    return prob, lane_mask_small, conf


def make_overlay(img_bgr, lane_mask_small, color=(0, 255, 0)):
    h, w = img_bgr.shape[:2]
    lane_full = cv2.resize(lane_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    lane_full = cv2.dilate(lane_full, kernel, iterations=2)
    colored = np.zeros_like(img_bgr)
    colored[lane_full > 0] = color
    result = cv2.addWeighted(img_bgr, 0.6, colored, 0.4, 0)
    return result, lane_full


def draw_hud(img, metrics=None, conf=None, frame_num=None):
    h, w = img.shape[:2]
    panel = img.copy()
    overlay = panel.copy()
    bar_h = 115 if metrics else 45
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, panel)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if conf is not None:
        cv2.putText(panel, f"Confidence: {conf:.4f}", (10, 28),
                    font, 0.65, (0, 255, 255), 2)
    if frame_num is not None:
        cv2.putText(panel, f"Frame: {frame_num}", (w - 170, 28),
                    font, 0.6, (255, 255, 255), 1)
    if metrics:
        items = [
            (f"IoU: {metrics['iou']:.3f}",            (0,   255, 100)),
            (f"Dice: {metrics['dice']:.3f}",           (0,   200, 255)),
            (f"Acc: {metrics['pixel_accuracy']:.3f}",  (255, 200,   0)),
            (f"Prec: {metrics['precision']:.3f}",      (255, 100, 100)),
            (f"Rec: {metrics['recall']:.3f}",          (200, 100, 255)),
        ]
        x = 10
        for text, color in items:
            cv2.putText(panel, text, (x, 75), font, 0.6, color, 2)
            x += 220
    return panel


def save_metrics_chart(metrics, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 Chart saved → {save_path}")


# ──────────────────────────────────────────────
# MODE 1: SINGLE IMAGE
# ──────────────────────────────────────────────
def run_image(path, gt_mask_path=None):
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot read: {path}")
        return

    prob, lane_small, conf = predict(img)
    result, _ = make_overlay(img, lane_small)

    metrics = None
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt       = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_small = cv2.resize(gt, (MODEL_W, MODEL_H)).astype(np.float32) / 255.0
        metrics  = compute_all_metrics(prob, gt_small)
        print_metrics(metrics, prefix="vs Ground Truth ")
    else:
        print("  ℹ️  No ground truth mask — confidence score only")
        print("     (Full metrics available only for dataset images)")

    result_hud = draw_hud(result, metrics=metrics, conf=conf)

    basename  = os.path.splitext(os.path.basename(path))[0]
    save_path = os.path.join(OUTPUT_DIR, "images", f"demo_{basename}.png")
    cv2.imwrite(save_path, result_hud)
    print(f"\n📸 Saved → {save_path}  |  Confidence: {conf:.4f}")

    if gt_mask_path and os.path.exists(gt_mask_path):
        gt_colored = cv2.cvtColor(
            cv2.resize(cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE),
                       (img.shape[1], img.shape[0])), cv2.COLOR_GRAY2BGR)
        comparison = np.hstack([img, gt_colored, result_hud])
        comp_path  = os.path.join(OUTPUT_DIR, "images", f"compare_{basename}.png")
        cv2.imwrite(comp_path, comparison)
        print(f"  Comparison → {comp_path}")

        chart_path = os.path.join(OUTPUT_DIR, "graphs", f"chart_{basename}.png")
        save_metrics_chart(metrics, f"Metrics: {basename}", chart_path)

    cv2.imshow("Lane Detection", result_hud)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# MODE 2: RANDOM
# ──────────────────────────────────────────────
def run_random():
    files    = sorted(set(os.listdir(IMG_DIR)) & set(os.listdir(MASK_DIR)))
    fname    = random.choice(files)
    img_path = os.path.join(IMG_DIR,  fname)
    msk_path = os.path.join(MASK_DIR, fname)
    print(f"Random: {fname}")
    run_image(img_path, gt_mask_path=msk_path)


# ──────────────────────────────────────────────
# MODE 3: VIDEO FILE
# ──────────────────────────────────────────────
def run_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Cannot open: {path}")
        return

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    oh, ow = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_path = os.path.join(OUTPUT_DIR, "videos", "video_output.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                               cap.get(cv2.CAP_PROP_FPS) or 25, (ow, oh))

    frame_count, conf_list = 0, []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        prob, lane_small, conf = predict(frame)
        result, _  = make_overlay(frame, lane_small)
        result_hud = draw_hud(result, conf=conf, frame_num=frame_count)
        writer.write(result_hud)
        conf_list.append(conf)
        frame_count += 1
        cv2.imshow("Video Lane Detection  (q to quit)", result_hud)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n🎬 Video saved → {out_path}")
    print(f"  Frames: {frame_count}  |  Mean conf: {np.mean(conf_list):.4f}")

    plt.figure(figsize=(12, 3))
    plt.plot(conf_list, linewidth=0.9, color="#2196F3")
    plt.axhline(np.mean(conf_list), color="red", linestyle="--",
                label=f"Mean {np.mean(conf_list):.4f}")
    plt.fill_between(range(len(conf_list)), conf_list, alpha=0.2, color="#2196F3")
    plt.title("Lane Detection Confidence per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Mean probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    graph_path = os.path.join(OUTPUT_DIR, "graphs", "video_confidence.png")
    plt.savefig(graph_path, dpi=150)
    plt.close()
    print(f"  Confidence plot → {graph_path}")


# ──────────────────────────────────────────────
# MODE 4: LIVE WEBCAM
# ──────────────────────────────────────────────
def run_live():
    print("\n📷 Starting live detection... Press Q to quit, S to save screenshot")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam found.")
        return

    frame_count, conf_list, screenshot_count = 0, [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        prob, lane_small, conf = predict(frame)
        result, _  = make_overlay(frame, lane_small)
        result_hud = draw_hud(result, conf=conf, frame_num=frame_count)
        cv2.putText(result_hud, "LIVE", (result_hud.shape[1] - 80, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        conf_list.append(conf)
        frame_count += 1
        cv2.imshow("Live Lane Detection  (Q=quit  S=screenshot)", result_hud)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts    = datetime.now().strftime("%H%M%S")
            spath = os.path.join(OUTPUT_DIR, "images", f"live_{ts}.png")
            cv2.imwrite(spath, result_hud)
            screenshot_count += 1
            print(f"📸 Screenshot → {spath}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Frames: {frame_count}  |  Mean conf: {np.mean(conf_list):.4f}")
    print(f"  Screenshots: {screenshot_count}")


# ──────────────────────────────────────────────
# MODE 5: FULL EVALUATION + ALL GRAPHS
# ──────────────────────────────────────────────
def run_full_evaluation(n=200):
    print(f"\n🔬 Running full evaluation on {n} samples...")

    files       = sorted(set(os.listdir(IMG_DIR)) & set(os.listdir(MASK_DIR)))[:n]
    metric_keys = ["iou", "dice", "pixel_accuracy", "precision", "recall"]
    all_metrics = {k: [] for k in metric_keys}
    csv_rows    = []

    for i, fname in enumerate(files):
        img  = cv2.imread(os.path.join(IMG_DIR,  fname))
        mask = cv2.imread(os.path.join(MASK_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        prob, _, _ = predict(img)
        gt_small   = cv2.resize(mask, (MODEL_W, MODEL_H)).astype(np.float32) / 255.0
        m          = compute_all_metrics(prob, gt_small)
        for k, v in m.items():
            all_metrics[k].append(v)
        csv_rows.append({"file": fname, **{k: f"{v:.4f}" for k, v in m.items()}})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)} done...")

    means = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    print_metrics(means, prefix=f"Final ({len(csv_rows)} samples) ")

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, "metrics", "eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file"] + metric_keys)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  📄 CSV → {csv_path}")

    # Graph 1: Mean metrics bar chart
    save_metrics_chart(
        means,
        f"Mean Evaluation Metrics  (n={len(csv_rows)})",
        os.path.join(OUTPUT_DIR, "graphs", "eval_metrics_summary.png")
    )

    # Graph 2: Per-sample IoU + Dice
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(all_metrics["iou"],  label="IoU",  alpha=0.7, linewidth=0.8, color="#2196F3")
    ax.plot(all_metrics["dice"], label="Dice", alpha=0.7, linewidth=0.8, color="#4CAF50")
    ax.axhline(means["iou"],  color="blue",  linestyle="--", linewidth=1,
               alpha=0.6, label=f"Mean IoU {means['iou']:.3f}")
    ax.axhline(means["dice"], color="green", linestyle="--", linewidth=1,
               alpha=0.6, label=f"Mean Dice {means['dice']:.3f}")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Score")
    ax.set_title("Per-sample IoU and Dice")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "graphs", "eval_per_sample.png"), dpi=150)
    plt.close()
    print(f"  📊 Per-sample → outputs/graphs/eval_per_sample.png")

    # Graph 3: All 5 metrics per sample
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    for ax, (k, vals), color in zip(axes, all_metrics.items(), colors):
        ax.plot(vals, linewidth=0.8, alpha=0.8, color=color)
        ax.axhline(np.mean(vals), color="red", linestyle="--", linewidth=1.2,
                   label=f"Mean {np.mean(vals):.3f}")
        ax.set_title(k.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Score")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].axis("off")
    plt.suptitle(f"All Metrics — {len(csv_rows)} Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "graphs", "eval_all_metrics.png"), dpi=150)
    plt.close()
    print(f"  📊 All metrics → outputs/graphs/eval_all_metrics.png")

    # Graph 4: Score distribution histograms
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, (k, vals), color in zip(axes, all_metrics.items(), colors):
        ax.hist(vals, bins=20, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(vals), color="red", linestyle="--",
                   label=f"Mean {np.mean(vals):.3f}")
        ax.set_title(k.replace("_", " ").title())
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    plt.suptitle("Score Distributions", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "graphs", "eval_distributions.png"), dpi=150)
    plt.close()
    print(f"  📊 Distributions → outputs/graphs/eval_distributions.png")

    print(f"\n✅ All done! Files saved:")
    print(f"   outputs/metrics/eval_results.csv")
    print(f"   outputs/graphs/eval_metrics_summary.png")
    print(f"   outputs/graphs/eval_per_sample.png")
    print(f"   outputs/graphs/eval_all_metrics.png")
    print(f"   outputs/graphs/eval_distributions.png")


# ──────────────────────────────────────────────
# MENU
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔════════════════════════════════════╗")
    print("║     Lane Detection Demo            ║")
    print("╠════════════════════════════════════╣")
    print("║  1. Image  (provide path)          ║")
    print("║  2. Random (from dataset)          ║")
    print("║  3. Video  (provide path)          ║")
    print("║  4. Live   (webcam)                ║")
    print("║  5. Evaluate All (graphs + CSV)    ║")
    print("╚════════════════════════════════════╝")

    choice = input("\nSelect [1/2/3/4/5]: ").strip()

    if choice == "1":
        img_p  = input("Image path: ").strip()
        fname  = os.path.basename(img_p)
        mask_p = os.path.join(MASK_DIR, os.path.splitext(fname)[0] + ".png")
        run_image(img_p, gt_mask_path=mask_p if os.path.exists(mask_p) else None)

    elif choice == "2":
        run_random()

    elif choice == "3":
        vid_p = input("Video path: ").strip()
        run_video(vid_p)

    elif choice == "4":
        run_live()

    elif choice == "5":
        n = input("How many samples? (default 200): ").strip()
        run_full_evaluation(n=int(n) if n.isdigit() else 200)

    else:
        print("Invalid choice.")