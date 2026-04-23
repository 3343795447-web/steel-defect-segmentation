import argparse
import glob
import os

import numpy as np


def fast_hist(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def per_class_iou(hist: np.ndarray) -> np.ndarray:
    denom = hist.sum(1) + hist.sum(0) - np.diag(hist)
    return np.diag(hist) / np.maximum(denom, 1)


def main():
    parser = argparse.ArgumentParser()
    # Keep defaults aligned with your improved calc_miou.py convention.
    parser.add_argument("--gt-dir", default="predict_output/test_ground_truths")
    parser.add_argument("--pred-dir", default="predict_output/baseline_predictions")
    parser.add_argument("--num-classes", type=int, default=4)
    args = parser.parse_args()

    gt_dir = args.gt_dir
    pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, "prediction_*.npy")))
    if not pred_paths:
        raise FileNotFoundError(f"No prediction_*.npy found in {args.pred_dir}")

    hist = np.zeros((args.num_classes, args.num_classes), dtype=np.float64)
    valid = 0

    for pred_path in pred_paths:
        stem = os.path.basename(pred_path).replace("prediction_", "").replace(".npy", "")
        # Match convention: prediction_{id}.npy vs ground_truth_{id}.npy
        gt_path = os.path.join(gt_dir, f"ground_truth_{stem}.npy")
        if not os.path.isfile(gt_path):
            continue

        pred = np.load(pred_path).astype(np.int64)
        gt = np.load(gt_path).astype(np.int64)
        if pred.shape != gt.shape:
            continue

        hist += fast_hist(gt.flatten(), pred.flatten(), args.num_classes)
        valid += 1

    if valid == 0:
        raise RuntimeError("No valid prediction/ground-truth pairs matched.")

    ious = per_class_iou(hist)
    print("Valid pairs:", valid)
    print("Per-class IoU:", np.round(ious, 4))
    print("mIoU(all classes):", round(float(np.nanmean(ious)), 4))
    print("mIoU(defect classes 1..N):", round(float(np.nanmean(ious[1:])), 4))


if __name__ == "__main__":
    main()

