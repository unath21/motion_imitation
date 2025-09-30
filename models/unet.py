""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM2d(nn.Module):
	def __init__(self, channels: int, cond_dim: int):
		super().__init__()
		self.channels = channels
		self.cond_dim = cond_dim
		self.mlp = nn.Linear(cond_dim, 2 * channels)
		nn.init.zeros_(self.mlp.weight)
		nn.init.zeros_(self.mlp.bias)

		with torch.no_grad():
			self.mlp.bias[:channels].fill_(1.0)

	def forward(self, x: torch.Tensor, cond: torch.Tensor):
		B, C, H, W = x.shape
		assert C == self.channels, f"channel mismatch: {C} != {self.channels}"
		gamma_beta = self.mlp(cond)  # (B, 2C)
		gamma, beta = gamma_beta.chunk(2, dim=-1)
		gamma = gamma.view(B, C, 1, 1)
		beta = beta.view(B, C, 1, 1)
		return gamma * x + beta

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, cond_dim=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.film = FiLM2d(out_channels, cond_dim) if cond_dim else None

    def forward(self, x, cond):
        x = double_conv(x)
        return self.film(x, cond)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)