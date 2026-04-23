import os
from PIL import Image
import numpy as np


def nonzero_ratio_gray_png(p: str) -> tuple[float, np.ndarray]:
    a = np.array(Image.open(p).convert("L"))
    # class id 0 is background; treat everything else as "non-zero"
    return float((a != 0).mean()), np.unique(a)[:10]


def main():
    self_viz = os.path.join(os.path.dirname(__file__), "quick_output", "viz")
    unet_viz = os.path.join(os.path.dirname(__file__), "quick_output_unet", "viz")

    candidates = []
    for idx in range(1, 100):
        gt = os.path.join(self_viz, f"gt_{idx:02d}.png")
        self_pred = os.path.join(self_viz, f"pred_{idx:02d}.png")
        unet_pred = os.path.join(unet_viz, f"pred_{idx:02d}.png")
        if not (os.path.exists(gt) and os.path.exists(self_pred) and os.path.exists(unet_pred)):
            continue

        gt_r, gt_vals = nonzero_ratio_gray_png(gt)
        self_r, self_vals = nonzero_ratio_gray_png(self_pred)
        unet_r, unet_vals = nonzero_ratio_gray_png(unet_pred)

        # Heuristic score: prefer cases where GT has defects,
        # SelfNet predicts almost background, and UNet predicts more non-background pixels.
        score = (gt_r) * 2.0 + (unet_r - self_r) * 5.0
        candidates.append((idx, gt_r, self_r, unet_r, score, gt_vals, self_vals, unet_vals))

    candidates.sort(key=lambda x: x[4], reverse=True)

    print("idx  gt_nonzero  self_pred_nonzero  unet_pred_nonzero  score")
    for c in candidates[:6]:
        idx, gt_r, self_r, unet_r, score, _, _, _ = c
        print(idx, round(gt_r, 4), round(self_r, 4), round(unet_r, 4), round(score, 4))

    print("\nTop recommended indices (2~3):", [c[0] for c in candidates[:3]])


if __name__ == "__main__":
    main()

