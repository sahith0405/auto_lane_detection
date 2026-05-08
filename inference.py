"""
Single-image inference with lane overlay and optional ground-truth comparison.

Usage:
    python inference.py                    # interactive prompt
    python inference.py --image path.jpg
    python inference.py --random
"""

import os
import cv2
import argparse
import random
import numpy as np
import torch

from model import SimpleUNet
from metrics import compute_all_metrics, print_metrics

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
IMG_DIR = "data/tusimple/processed/images"
MASK_DIR = "data/tusimple/processed/masks"
MODEL_PATH = "checkpoints/model_best.pth"
OUTPUT_DIR = "outputs"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
model = SimpleUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# ──────────────────────────────────────────────
# LANE POST-PROCESSING  (smooth lines)
# ──────────────────────────────────────────────
#def refine_lanes(prob_map: np.ndarray, threshold=0.5) -> np.ndarray:
def refine_lanes(prob_map: np.ndarray, threshold=0.3) -> np.ndarray:
    """
    Convert raw probability map to clean lane lines via:
    1. Threshold → binary mask
    2. Morphological closing to fill small gaps
    3. Hough line fitting for smooth, straight-ish lines
    Returns a uint8 mask (0 / 255) at the same resolution as prob_map.
    """
    binary = (prob_map > threshold).astype(np.uint8) * 255

    # Close small gaps in the lane markings
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Remove tiny noise blobs
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # Hough lines for smooth rendering
    edges = cv2.Canny(binary, 30, 100)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=30,
        minLineLength=20, maxLineGap=40
    )

    clean = np.zeros_like(binary)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(clean, (x1, y1), (x2, y2), 255, thickness=3)

    # Fall back to morphological result if Hough found nothing
    if clean.sum() == 0:
        clean = binary

    return clean


# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────
def run_inference(img_path: str):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"ERROR: Cannot read image: {img_path}")
        return

    h, w = img_bgr.shape[:2]

    # Preprocess
    inp = cv2.resize(img_bgr, (256, 128)).astype(np.float32) / 255.0
    inp_t = torch.tensor(inp).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_map = model(inp_t)[0][0].cpu().numpy()  # (128, 256) in [0,1]

    confidence = float(np.mean(pred_map))

    # Smooth lane mask at model resolution
    lane_mask_small = refine_lanes(pred_map)

    # Scale back to original image size
    lane_mask = cv2.resize(lane_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # ── Overlay: green lanes on original image ──
    overlay = img_bgr.copy()
    green_layer = np.zeros_like(overlay)
    green_layer[lane_mask > 0] = (0, 255, 0)
    result = cv2.addWeighted(overlay, 0.75, green_layer, 0.55, 0)

    # ── Save ──
    basename = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(OUTPUT_DIR, f"inference_{basename}.png")
    cv2.imwrite(save_path, result)

    print(f"\n📸 Inference result")
    print(f"  Image      : {img_path}")
    print(f"  Confidence : {confidence:.4f}")
    print(f"  Saved to   : {save_path}")

    # ── Optional ground-truth comparison ──
    mask_path = os.path.join(MASK_DIR, os.path.basename(img_path).replace(".jpg", ".png"))
    if not os.path.exists(mask_path):
        # Try same extension
        mask_path = os.path.join(MASK_DIR, os.path.basename(img_path))

    if os.path.exists(mask_path):
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_small = cv2.resize(gt_mask, (256, 128)).astype(np.float32) / 255.0
        metrics = compute_all_metrics(pred_map, gt_small)
        print_metrics(metrics, prefix="vs Ground Truth ")
    else:
        print("  (No ground-truth mask found — skipping metric comparison)")

    # Show
    cv2.imshow("Lane Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--random", action="store_true", help="Pick a random image from dataset")
    args = parser.parse_args()

    if args.image:
        run_inference(args.image)

    elif args.random:
        files = sorted(set(os.listdir(IMG_DIR)) & set(os.listdir(MASK_DIR)))
        run_inference(os.path.join(IMG_DIR, random.choice(files)))

    else:
        choice = input("Enter image path OR type 'random': ").strip()
        if choice.lower() == "random":
            files = sorted(set(os.listdir(IMG_DIR)) & set(os.listdir(MASK_DIR)))
            run_inference(os.path.join(IMG_DIR, random.choice(files)))
        else:
            run_inference(choice)