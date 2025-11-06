import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.models.autoencoders.vae import Encoder

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


class ResNetBlock2DConditional(nn.Module):
	def __init__(self, in_channels, out_channels, cond_dim=None, stride=1):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		# First conv block
		self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
							   stride=stride, padding=1, bias=False)

		# Second conv block
		self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
							   stride=1, padding=1, bias=False)

		# FiLM conditioning
		self.film1 = FiLM2d(out_channels, cond_dim) if cond_dim else None
		self.film2 = FiLM2d(out_channels, cond_dim) if cond_dim else None

		if stride != 1 or in_channels != out_channels:
			self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1,
								  stride=stride, bias=False)
		else:
			self.skip = nn.Identity()

		self.activation = nn.SiLU()

	def forward(self, x, cond=None):
		skip = self.skip(x)

		# First conv
		h = self.norm1(x)
		h = self.activation(h)
		h = self.conv1(h)
		if self.film1 is not None and cond is not None:
			h = self.film1(h, cond)

		# Second conv
		h = self.norm2(h)
		h = self.activation(h)
		h = self.conv2(h)
		if self.film2 is not None and cond is not None:
			h = self.film2(h, cond)

		return h + skip


class DownBlock2DConditional(nn.Module):
	def __init__(self, in_channels, out_channels, num_layers=2, cond_dim=None):
		super().__init__()

		layers = []
		layers.append(ResNetBlock2DConditional(in_channels, out_channels, cond_dim, stride=2))

		for _ in range(num_layers - 1):
			layers.append(ResNetBlock2DConditional(out_channels, out_channels, cond_dim))

		self.layers = nn.ModuleList(layers)

	def forward(self, x, cond=None):
		for layer in self.layers:
			x = layer(x, cond)
		return x


class UpBlock2DConditional(nn.Module):
	def __init__(self, in_channels, out_channels, num_layers=2, cond_dim=None, skip_channels=None):
		super().__init__()

		self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

		if skip_channels is None:
			skip_channels = out_channels

		layers = []
		first_in = out_channels + skip_channels
		layers.append(ResNetBlock2DConditional(first_in, out_channels, cond_dim))

		for _ in range(num_layers - 1):
			layers.append(ResNetBlock2DConditional(out_channels, out_channels, cond_dim))

		self.layers = nn.ModuleList(layers)

	def forward(self, x, skip=None, cond=None):
		# Upsample
		x = self.upsample(x)

		if skip is not None:
			dh = skip.shape[2] - x.shape[2]
			dw = skip.shape[3] - x.shape[3]
			pad = [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2]
			if any(p != 0 for p in pad):
				x = F.pad(x, pad)
			x = torch.cat([skip, x], dim=1)

		for layer in self.layers:
			x = layer(x, cond)

		return x


class UNet2DConditionModel(nn.Module):
	def __init__(self, in_channels=3, out_channels=3, cond_dim=64,
				 block_out_channels=(64, 128, 256), layers_per_block=2):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.cond_dim = cond_dim

		# Initial convolution
		self.conv_in = nn.Conv2d(in_channels, block_out_channels[0],
								 kernel_size=3, padding=1)

		self.down_blocks = nn.ModuleList()
		in_ch = block_out_channels[0]
		for out_ch in block_out_channels:
			self.down_blocks.append(
				DownBlock2DConditional(in_ch, out_ch, layers_per_block, cond_dim)
			)
			in_ch = out_ch

		self.up_blocks = nn.ModuleList()
		reversed_channels = list(reversed(block_out_channels))
		in_ch = block_out_channels[-1]
		for i, out_ch in enumerate(reversed_channels):
			self.up_blocks.append(
				UpBlock2DConditional(in_ch, out_ch, layers_per_block, cond_dim, skip_channels=out_ch)
			)
			in_ch = out_ch

		self.conv_out = nn.Sequential(
			nn.ConvTranspose2d(block_out_channels[0], block_out_channels[0], kernel_size=2, stride=2),

			nn.GroupNorm(min(32, block_out_channels[0]), block_out_channels[0]),
			nn.SiLU(),
			nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
		)

	def forward(self, sample, cond=None, return_dict=False):
		x = self.conv_in(sample)

		skips = []
		for down_block in self.down_blocks:
			x = down_block(x, cond)
			skips.append(x)

		for idx, up_block in enumerate(self.up_blocks):
			skip = skips.pop() if len(skips) > 0 else None
			x = up_block(x, skip=skip, cond=cond)

		x = self.conv_out(x)

		if return_dict:
			return {"sample": x}
		return (x,)
