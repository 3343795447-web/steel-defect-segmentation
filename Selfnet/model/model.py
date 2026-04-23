"""
SelfNet: a lightweight encoder-decoder segmentation model.

Compared to the baseline UNet:
- Fewer down/up stages (shallower), fewer parameters
- Uses transposed convolution for upsampling

Outputs:
- Raw logits with shape [B, num_classes, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x2 is the skip connection from encoder.
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class SelfNet(nn.Module):  # 类名保持大写
    def __init__(self, in_channels=1, num_classes=2, base_c=46):
        """
        Args:
            in_channels: number of input channels (3 for RGB images)
            num_classes: number of segmentation classes
            base_c: base channel width (controls model capacity)
        """
        super(SelfNet, self).__init__()
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.up1 = Up(base_c * 4, base_c * 2)
        self.up2 = Up(base_c * 2, base_c)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # Decoder with skip connections
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.out_conv(x)
