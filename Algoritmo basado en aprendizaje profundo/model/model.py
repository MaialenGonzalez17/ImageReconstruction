import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True))

        # Decoder
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, padding=1), nn.ReLU(True))
        self.up1 = nn.Upsample(scale_factor=2)
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128 + 128, 64, 3, padding=1), nn.ReLU(True))
        self.up2 = nn.Upsample(scale_factor=2)
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(64 + 64, 64, 3, padding=1), nn.Sigmoid())

        self.residual = nn.Conv2d(64, 3, kernel_size=3, padding=1)  # residual output
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x3 = self.enc2(x2)
        x4 = self.pool2(x3)
        x5 = self.enc3(x4)

        d1 = self.dec1(x5)
        d1_up = self.up1(d1)
        d2 = self.dec2(torch.cat([d1_up, x3], dim=1))
        d2_up = self.up2(d2)
        d3 = self.dec3(torch.cat([d2_up, x1], dim=1))

        residual = self.tanh(self.residual(d3))
        out = x + residual
        return out
