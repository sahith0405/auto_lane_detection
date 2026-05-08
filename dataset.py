import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TuSimpleDataset(Dataset):
    """
    Loads paired (image, mask) samples from TuSimple processed directory.

    Args:
        img_dir  : folder with images  (.png)
        mask_dir : folder with masks   (.png, single-channel)
        augment  : apply random horizontal flip + brightness jitter
    """

    def __init__(self, img_dir, mask_dir, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment

        # Build list of files that exist in BOTH directories
        img_files = set(os.listdir(img_dir))
        mask_files = set(os.listdir(mask_dir))
        common = sorted(img_files & mask_files)

        if len(common) == 0:
            raise RuntimeError(
                f"No matching files found between:\n  {img_dir}\n  {mask_dir}\n"
                "Make sure preprocess_tusimple.py has been run."
            )

        self.files = common
        print(f"[Dataset] Found {len(self.files)} paired samples. Augment={augment}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        # Resize to model input size
        image = cv2.resize(image, (256, 128), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (256, 128), interpolation=cv2.INTER_NEAREST)

        # Augmentation (training only)
        if self.augment:
            if np.random.rand() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)

            # Brightness / contrast jitter
            alpha = np.random.uniform(0.8, 1.2)   # contrast
            beta = np.random.randint(-20, 20)      # brightness
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # To tensor
        image = torch.tensor(image).permute(2, 0, 1)   # (3, H, W)
        mask = torch.tensor(mask).unsqueeze(0)          # (1, H, W)

        return image, mask
count = sum(len(files) for _, _, files in os.walk("clips"))
print(count)