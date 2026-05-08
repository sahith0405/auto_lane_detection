"""
Train SimpleUNet on TuSimple processed data.

Usage:
    python train.py

Outputs:
    checkpoints/model_best.pth   ← best val IoU
    checkpoints/model_last.pth   ← last epoch
    outputs/loss_graph.png
    outputs/metrics_graph.png
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import TuSimpleDataset
from model import SimpleUNet
from metrics import iou_score, dice_score

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
IMG_DIR = "data/tusimple/processed/images"
MASK_DIR = "data/tusimple/processed/masks"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"

EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-3
VAL_SPLIT = 0.1        # 10 % of data held out for validation
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")

# ──────────────────────────────────────────────
# COMBINED BCE + DICE LOSS
# ──────────────────────────────────────────────
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_w = bce_weight

    def forward(self, pred, target):
        bce = self.bce(pred, target)

        inter = (pred * target).sum(dim=(1, 2, 3))
        dice = 1.0 - (2 * inter + 1e-6) / (
            pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
        )
        dice = dice.mean()

        return self.bce_w * bce + (1 - self.bce_w) * dice

# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────
full_dataset = TuSimpleDataset(IMG_DIR, MASK_DIR, augment=True)

val_size = max(1, int(len(full_dataset) * VAL_SPLIT))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# Val set should not be augmented — wrap it with augment=False dataset
val_ds_clean = TuSimpleDataset(IMG_DIR, MASK_DIR, augment=False)
val_indices = val_ds.indices
from torch.utils.data import Subset
val_ds_clean = Subset(val_ds_clean, val_indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader = DataLoader(val_ds_clean, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)

print(f"Train: {train_size}  |  Val: {val_size}")

# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
model = SimpleUNet().to(DEVICE)
criterion = BCEDiceLoss(bce_weight=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

# ──────────────────────────────────────────────
# HISTORY
# ──────────────────────────────────────────────
history = {
    "train_loss": [], "val_loss": [],
    "val_iou": [], "val_dice": []
}
best_iou = 0.0

# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    # ── Train ──
    model.train()
    train_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [train]", leave=False)
    for imgs, masks in loop:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    train_loss /= len(train_loader)

    # ── Validate ──
    model.eval()
    val_loss = 0.0
    ious, dices = [], []

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            val_loss += criterion(preds, masks).item()

            for p, m in zip(preds.cpu().numpy(), masks.cpu().numpy()):
                ious.append(iou_score(p[0], m[0]))
                dices.append(dice_score(p[0], m[0]))

    val_loss /= len(val_loader)
    mean_iou = float(np.mean(ious))
    mean_dice = float(np.mean(dices))

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_iou"].append(mean_iou)
    history["val_dice"].append(mean_dice)

    print(
        f"Epoch {epoch:02d}/{EPOCHS}  "
        f"TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  "
        f"IoU={mean_iou:.4f}  Dice={mean_dice:.4f}"
    )

    # ── Scheduler ──
    scheduler.step(mean_iou)

    # ── Save best ──
    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_best.pth"))
        print(f"  ✔ New best model saved  (IoU={best_iou:.4f})")

# ── Save last ──
torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_last.pth"))
# Also save as model.pth for backward compat with other scripts
torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model.pth"))

# ──────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────
epochs_range = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs_range, history["train_loss"], label="Train Loss")
axes[0].plot(epochs_range, history["val_loss"], label="Val Loss")
axes[0].set_title("Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs_range, history["val_iou"], label="Val IoU")
axes[1].plot(epochs_range, history["val_dice"], label="Val Dice")
axes[1].set_title("Validation Metrics")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
plt.close()
print(f"\nTraining curves saved → outputs/training_curves.png")
print(f"Best Val IoU: {best_iou:.4f}")
print("Training DONE ✔")