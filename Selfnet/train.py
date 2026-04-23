"""
Training entry for a 4-class semantic segmentation task.

This script assumes a "VOC-like" folder layout (hard-coded below):
- Images:  dataset/JPEGImages/*.jpg
- Masks:   dataset/Annotations/*.png

Mask encoding:
- Each pixel value is treated as a class id in {0,1,2,3} where 0 is typically background.

Outputs:
- Saves the best checkpoint to: ./{UNet_pth|SelfNet_pth}/best_model.pth
- Also saves periodic checkpoints every 5 epochs.

Note:
- The default num_epochs=30000 is very large; adjust for your compute budget.
"""

import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


from model.unet import UNet
from model.model import SelfNet


def calculate_iou(outputs, labels, num_classes):
    """
    Compute per-class IoU and mean IoU on a batch.

    Args:
        outputs: logits with shape [B, C, H, W]
        labels:  integer mask with shape [B, H, W]
        num_classes: number of classes C
    """
    iou_per_class = np.zeros(num_classes)
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)

    outputs = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    for cls in range(num_classes):
        intersection[cls] = np.sum((outputs == cls) & (labels == cls))
        union[cls] = np.sum((outputs == cls) | (labels == cls))

    for cls in range(num_classes):
        if union[cls] > 0:
            iou_per_class[cls] = intersection[cls] / union[cls]

    mIoU = np.mean(iou_per_class)
    return iou_per_class, mIoU

def evaluate_model(model, dataloader, num_classes):
    """
    Lightweight evaluation loop that accumulates IoU over batches.
    """
    model.eval()
    all_iou = np.zeros(num_classes)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # labels from this dataset are [B, H, W]; if you change dataset to return [B, 1, H, W],
            # then squeeze(1) becomes necessary.
            iou_per_class, _ = calculate_iou(outputs, labels.long(), num_classes)
            all_iou += iou_per_class

    mIoU = np.mean(all_iou)
    return all_iou, mIoU

class CustomDataset(torch.utils.data.Dataset):
    """
    Minimal dataset that pairs each image with a same-name mask.

    Example:
        image: 000123.jpg
        mask:  000123.png
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = os.path.join(self.label_dir, os.path.basename(image_path).replace('.jpg', '.png'))
    
        # Image is RGB; mask is a single-channel image whose pixel values represent class ids.
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
    
        # Convert mask to integer ids. This assumes values are already in [0, num_classes-1].
        label = np.array(label).astype(np.uint8)
    
        if self.transform:
            image = self.transform(image)
            label = torch.tensor(label, dtype=torch.long)
    
        return image, label


if __name__ == '__main__':
    # Training hyper-parameters.
    num_epochs = 30000
    batch_size = 64 #32
    
    learning_rate = 0.0001
    best_loss = float('inf')

    # Dataset paths (expected to exist on disk).
    image_dir = 'dataset/JPEGImages'
    label_dir = 'dataset/Annotations'
    dataset = CustomDataset(image_dir, label_dir, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Choose which model to train.
    model_select = 'SelfNet'   # UNet,SelfNet
    
    if model_select == 'UNet':
        model = UNet(in_channels=3, num_classes=4).to(device)
    if model_select == 'SelfNet':
        model = SelfNet(in_channels=3, num_classes=4).to(device)
    save_pth_path = model_select+'_pth'
    print('model train:',model_select,' save in:',save_pth_path)

    # Quick sanity info: parameter count.
    print('---------------------------------')
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())/1000000}')
    print('---------------------------------')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            # CrossEntropyLoss expects:
            # - outputs: [B, C, H, W]
            # - labels:  [B, H, W] (dtype long)
            loss = criterion(outputs, labels.long())

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), './{}/best_model.pth'.format(save_pth_path))
            print('Best model saved!')
            
        if (epoch + 1) % 5 == 0:
            best_loss = avg_loss
            torch.save(model.state_dict(), './{}/Epoch_{}.pth'.format(save_pth_path,epoch + 1))

    print('Training completed!')







