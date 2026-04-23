import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model.unet import UNet


def main():
    parser = argparse.ArgumentParser()
    # Keep defaults aligned with your improved pipeline:
    # input_dir = ./dataset/images_test/
    parser.add_argument("--input-dir", default="dataset/images_test")
    parser.add_argument("--ckpt", default="UNet_pth/best_model.pth")
    parser.add_argument("--num-classes", type=int, default=4)
    # UNet baseline output dir, consistent with calc_miou.py in your project
    parser.add_argument("--out-dir", default="predict_output/baseline_predictions")
    parser.add_argument("--save-png", action="store_true", help="also save prediction masks as png")
    args = parser.parse_args()

    image_dir = args.input_dir
    os.makedirs(args.out_dir, exist_ok=True)
    png_dir = os.path.join(args.out_dir, "png_masks")
    if args.save_png:
        os.makedirs(png_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    to_tensor = transforms.ToTensor()
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.*")))
    print(f"Predicting on {len(image_paths)} images, device={device}")

    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            x = to_tensor(image).unsqueeze(0).to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            stem = os.path.splitext(os.path.basename(image_path))[0]
            npy_path = os.path.join(args.out_dir, f"prediction_{stem}.npy")
            np.save(npy_path, pred)

            if args.save_png:
                Image.fromarray(pred).save(os.path.join(png_dir, f"prediction_{stem}.png"))

    print(f"Done. Saved predictions to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

