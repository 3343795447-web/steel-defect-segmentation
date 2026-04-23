import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model.unet import UNet


class SegDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.mask_dir = mask_dir
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(self.mask_dir, f"{stem}.png")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.to_tensor(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()
        return image, mask


def main():
    parser = argparse.ArgumentParser()
    # Keep defaults aligned with your improved pipeline:
    # dataset/JPEGImages + dataset/Annotations
    parser.add_argument("--image-dir", default="dataset/JPEGImages")
    parser.add_argument("--mask-dir", default="dataset/Annotations")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", default="UNet_pth")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    image_dir = args.image_dir
    mask_dir = args.mask_dir
    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Dataset folders not found:\n{image_dir}\n{mask_dir}")

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SegDataset(image_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = UNet(in_channels=3, num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    print(f"Training UNet on {len(dataset)} images, device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        avg_loss = running_loss / max(len(loader), 1)
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print("Saved best_model.pth")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"Epoch_{epoch}.pth"))

    print("Training done.")


if __name__ == "__main__":
    main()

