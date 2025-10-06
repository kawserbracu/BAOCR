from __future__ import annotations
import torch
import torch.nn as nn

class MNV3Seg(nn.Module):
    """MobileNetV3 backbone + lightweight decoder to 1-channel segmentation map.
    Used for custom DB-like detector.
    """
    def __init__(self, backbone: str = 'large', pretrained: bool = True):
        super().__init__()
        from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
        if backbone == 'large':
            self.backbone = mobilenet_v3_large(weights='DEFAULT' if pretrained else None).features
            ch = 960
        else:
            self.backbone = mobilenet_v3_small(weights='DEFAULT' if pretrained else None).features
            ch = 576
        self.decoder = nn.Sequential(
            nn.Conv2d(ch, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, kernel_size=1),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        out = self.decoder(feats)
        return out  # (N,1,H',W')
