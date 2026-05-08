import numpy as np


def iou_score(pred, target, threshold=0.5):
    pred = (pred > threshold).astype(np.uint8)
    target = (target > threshold).astype(np.uint8)

    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    return inter / (union + 1e-6)


def dice_score(pred, target, threshold=0.5):
    pred = (pred > threshold).astype(np.uint8)
    target = (target > threshold).astype(np.uint8)

    inter = np.sum(pred * target)
    return (2.0 * inter) / (np.sum(pred) + np.sum(target) + 1e-6)


def pixel_accuracy(pred, target, threshold=0.5):
    pred = (pred > threshold).astype(np.uint8)
    target = (target > threshold).astype(np.uint8)

    return (pred == target).mean()


def precision_score(pred, target, threshold=0.5):
    pred = (pred > threshold).astype(np.uint8)
    target = (target > threshold).astype(np.uint8)

    tp = np.logical_and(pred == 1, target == 1).sum()
    fp = np.logical_and(pred == 1, target == 0).sum()

    return tp / (tp + fp + 1e-6)


def recall_score(pred, target, threshold=0.5):
    pred = (pred > threshold).astype(np.uint8)
    target = (target > threshold).astype(np.uint8)

    tp = np.logical_and(pred == 1, target == 1).sum()
    fn = np.logical_and(pred == 0, target == 1).sum()

    return tp / (tp + fn + 1e-6)


def compute_all_metrics(pred, target, threshold=0.5):
    """Compute all metrics and return as dict."""
    return {
        "iou": iou_score(pred, target, threshold),
        "dice": dice_score(pred, target, threshold),
        "pixel_accuracy": pixel_accuracy(pred, target, threshold),
        "precision": precision_score(pred, target, threshold),
        "recall": recall_score(pred, target, threshold),
    }


def print_metrics(metrics: dict, prefix=""):
    print(f"\n{prefix}📊 Evaluation Metrics")
    print(f"  IoU (Jaccard):   {metrics['iou']:.4f}")
    print(f"  Dice (F1):       {metrics['dice']:.4f}")
    print(f"  Pixel Accuracy:  {metrics['pixel_accuracy']:.4f}")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")