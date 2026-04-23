"""
Quick smoke test runner (small subset training + evaluation + visualization).

Why this exists:
- The original scripts assume a local `dataset/` folder layout.
- Your repo currently contains data under `NEU_Seg-main/`.
- This script trains for a few epochs on a small random subset so you can:
  - verify the pipeline works end-to-end
  - quickly compare SelfNet vs UNet under identical settings
  - generate a few (img/gt/pred) triplets for presentation

Outputs:
- {out_dir}/{Model}_quick.pth
- {out_dir}/viz/img_XX.jpg, gt_XX.png, pred_XX.png
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model.model import SelfNet
from model.unet import UNet


@dataclass(frozen=True)
class Sample:
    image_path: str
    mask_path: str


class NEUSegDataset(Dataset):
    def __init__(self, samples: List[Sample], image_transform=None):
        self.samples = samples
        self.image_transform = image_transform or transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        mask = Image.open(s.mask_path).convert("L")

        image_t = self.image_transform(image)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_t = torch.from_numpy(mask_np).long()
        # Returned shapes:
        # - image_t: [3, H, W]
        # - mask_t:  [H, W] (class ids)
        return image_t, mask_t


def list_pairs(images_dir: str, masks_dir: str) -> List[Sample]:
    """
    Pairs images and masks by filename stem.
    Image can be .jpg/.png/...; mask is expected to be .png with same stem.
    """
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in os.listdir(images_dir) if os.path.splitext(f.lower())[1] in valid_ext
    ]
    samples: List[Sample] = []
    for name in sorted(image_files):
        stem, _ = os.path.splitext(name)
        mask_path = os.path.join(masks_dir, f"{stem}.png")
        img_path = os.path.join(images_dir, name)
        if os.path.isfile(mask_path):
            samples.append(Sample(image_path=img_path, mask_path=mask_path))
    return samples


def fast_hist(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Confusion matrix for segmentation (flattened arrays)."""
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def iou_from_hist(hist: np.ndarray) -> np.ndarray:
    """Per-class IoU from confusion matrix."""
    denom = (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return np.diag(hist) / np.maximum(denom, 1)


@torch.no_grad()
def evaluate_miou(model: torch.nn.Module, loader: DataLoader, num_classes: int, device: torch.device):
    """
    Computes per-class IoU and mIoU on the provided loader.
    mIoU here is a plain mean over all classes (including background).
    """
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        preds_np = preds.cpu().numpy().astype(np.int64)
        masks_np = masks.cpu().numpy().astype(np.int64)
        for p, m in zip(preds_np, masks_np):
            hist += fast_hist(m.flatten(), p.flatten(), num_classes)
    iou = iou_from_hist(hist)
    return iou, float(np.nanmean(iou))


@torch.no_grad()
def save_visual_examples(
    model: torch.nn.Module,
    samples: List[Sample],
    out_dir: str,
    device: torch.device,
    num_examples: int,
):
    """
    Exports a few samples as:
    - img_XX.jpg: original image
    - gt_XX.png:  ground truth mask (class ids)
    - pred_XX.png: predicted mask (class ids)
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    tfm = transforms.ToTensor()
    chosen = samples[:num_examples]
    for i, s in enumerate(chosen, start=1):
        img = Image.open(s.image_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        Image.fromarray(pred).save(os.path.join(out_dir, f"pred_{i:02d}.png"))

        gt = Image.open(s.mask_path).convert("L")
        gt.save(os.path.join(out_dir, f"gt_{i:02d}.png"))
        img.save(os.path.join(out_dir, f"img_{i:02d}.jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["SelfNet", "UNet"], default="SelfNet")
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--train-n", type=int, default=80, help="number of training samples to use")
    ap.add_argument("--val-n", type=int, default=40, help="number of validation samples to use")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="quick_output")
    args = ap.parse_args()

    # Fix randomness so runs are comparable between SelfNet and UNet.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base = os.path.join(os.path.dirname(__file__), "NEU_Seg-main")
    train_images = os.path.join(base, "images", "training")
    train_masks = os.path.join(base, "annotations", "training")
    if not (os.path.isdir(train_images) and os.path.isdir(train_masks)):
        raise SystemExit(f"NEU-Seg paths not found under: {base}")

    all_samples = list_pairs(train_images, train_masks)
    if len(all_samples) < (args.train_n + args.val_n):
        raise SystemExit(f"Not enough paired samples. Found {len(all_samples)}")

    random.shuffle(all_samples)
    train_samples = all_samples[: args.train_n]
    val_samples = all_samples[args.train_n : args.train_n + args.val_n]

    device = torch.device(args.device)
    if args.model == "UNet":
        model = UNet(in_channels=3, num_classes=args.num_classes).to(device)
    else:
        model = SelfNet(in_channels=3, num_classes=args.num_classes).to(device)

    train_ds = NEUSegDataset(train_samples)
    val_ds = NEUSegDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        for images, masks in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            # CrossEntropyLoss expects masks as [B, H, W] of integer class ids.
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        iou, miou = evaluate_miou(model, val_loader, args.num_classes, device)
        print(f"[epoch {epoch}] loss={mean_loss:.4f} val_mIoU={miou:.4f} per_class_iou={np.round(iou, 4)}")

    torch.save(model.state_dict(), os.path.join(args.out_dir, f"{args.model}_quick.pth"))
    save_visual_examples(
        model=model,
        samples=val_samples,
        out_dir=os.path.join(args.out_dir, "viz"),
        device=device,
        num_examples=min(6, len(val_samples)),
    )
    print(f"Saved weights and visualizations to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

