"""
Preprocess TuSimple dataset into (image, mask) pairs.

Usage:
    python preprocess_tusimple.py

Directory structure expected:
    data/tusimple/train_set/
        label_data_0601.json
        label_data_0531.json
        label_data_0313.json
        clips/...

Output:
    data/tusimple/processed/images/   ← original frames
    data/tusimple/processed/masks/    ← binary lane masks
"""

import os
import json
import cv2
import numpy as np

DATA_ROOT = "data/tusimple/train_set"
OUTPUT_DIR = "data/tusimple/processed"

os.makedirs(f"{OUTPUT_DIR}/masks", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

LABEL_FILES = [
    os.path.join(DATA_ROOT, "label_data_0601.json"),
    os.path.join(DATA_ROOT, "label_data_0531.json"),
    os.path.join(DATA_ROOT, "label_data_0313.json"),
]


def load_json():
    data = []
    for f in LABEL_FILES:
        if os.path.exists(f):
            with open(f, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            print(f"  Loaded: {f}")
        else:
            print(f"  Missing (skipped): {f}")
    return data


def create_mask(h_samples, lanes, shape):
    """
    Draw lane lines onto a blank mask.
    Uses thick lines (thickness=8) so the mask has enough positive pixels
    for the model to learn from.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for lane in lanes:
        points = [
            (int(x), int(y))
            for x, y in zip(lane, h_samples)
            if x != -2
        ]

        for i in range(len(points) - 1):
            cv2.line(mask, points[i], points[i + 1], color=255, thickness=8)

    return mask


def preprocess():
    print("Loading annotation files …")
    data = load_json()
    print(f"Total samples: {len(data)}\n")

    saved = 0
    skipped = 0

    for i, item in enumerate(data):
        img_path = os.path.join(DATA_ROOT, item["raw_file"])

        if not os.path.exists(img_path):
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]
        mask = create_mask(item["h_samples"], item["lanes"], (h, w))

        # Flatten path into a safe filename: clips/0313-1/0/20.jpg → clips_0313-1_0_20.png
        name = item["raw_file"].replace("/", "_").replace(".jpg", ".png")

        cv2.imwrite(os.path.join(OUTPUT_DIR, "images", name), img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "masks", name), mask)

        saved += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(data)} …")

    print(f"\nSaved : {saved}")
    print(f"Skipped: {skipped}")
    print("DONE ✔")


if __name__ == "__main__":
    preprocess()