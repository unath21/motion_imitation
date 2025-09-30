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

class ResNetBlock2D(nn.Module):
	"""ResNet block with FiLM conditioning"""
	def __init__(self, in_channels, out_channels, cond_dim=None, stride=1):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

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

		self.activation = nn.SiLU()

	def forward(self, x, cond=None):
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

		# FiLM conditioning after second conv
		if self.film2 is not None and cond is not None:
			h = self.film2(h, cond)

		return h

class DownBlock2D(nn.Module):
	"""Downsampling block with ResNet blocks and FiLM"""
	def __init__(self, in_channels, out_channels, num_layers=2, cond_dim=None):
		super().__init__()

		layers = []
		# First layer with downsampling
		layers.append(ResNetBlock2D(in_channels, out_channels, cond_dim, stride=2))

		# Additional layers
		for _ in range(num_layers - 1):
			layers.append(ResNetBlock2D(out_channels, out_channels, cond_dim))

		self.layers = nn.ModuleList(layers)

	def forward(self, x, cond=None):
		for layer in self.layers:
			x = layer(x, cond)
		return x

class UpBlock2D(nn.Module):
	"""Upsampling block with ResNet blocks and FiLM"""
	def __init__(self, in_channels, out_channels, num_layers=2, cond_dim=None):
		super().__init__()

		# Upsample layer
		self.upsample = nn.ConvTranspose2d(in_channels, in_channels,
										  kernel_size=2, stride=2)
		layers = []
		# Additional layers
		layers.append(ResNetBlock2D(in_channels, out_channels, cond_dim))
		for _ in range(num_layers - 1):
			layers.append(ResNetBlock2D(out_channels, out_channels, cond_dim))

		self.layers = nn.ModuleList(layers)


	def forward(self, x, cond=None):
		# Upsample
		x = self.upsample(x)

		# Apply ResNet blocks
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

		# ---------------- Encoder ----------------
		self.down_blocks = nn.ModuleList()
		in_ch = block_out_channels[0]

		for out_ch in block_out_channels:
			self.down_blocks.append(
				DownBlock2D(in_ch, out_ch, layers_per_block, cond_dim)
			)
			in_ch = out_ch

		# ---------------- Decoder ----------------
		self.up_blocks = nn.ModuleList()
		reversed_channels = list(reversed(block_out_channels))

		for out_ch in reversed_channels:
			self.up_blocks.append(
				UpBlock2D(in_ch, out_ch, layers_per_block, cond_dim)
			)
			in_ch = out_ch

		# Output convolution
		self.conv_out = nn.Sequential(
			nn.GroupNorm(min(32, in_ch), in_ch),
			nn.SiLU(),
			nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)
		)

	def forward(self, sample, cond=None, return_dict=False):
		# Initial conv
		x = self.conv_in(sample)

		# Encoder
		for down_block in self.down_blocks:
			x = down_block(x, cond)

		# Decoder (no skips)
		for up_block in self.up_blocks:
			x = up_block(x, cond)

		# Output
		x = self.conv_out(x)

		if return_dict:
			return {"sample": x}
		return (x,)


class Autoencoder(nn.Module):
	def __init__(self, in_channels=3, out_channels=3, z_channels=128):
		super().__init__()
		self.encoder = Encoder(
			in_channels=in_channels,
			out_channels=z_channels,
			down_block_types=("DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"),
			block_out_channels=(64, 128, 256),
			layers_per_block=1,
			act_fn='silu',
			double_z=False,
			norm_num_groups=32,
			mid_block_add_attention=False,
		)

		self.global_pool = nn.AdaptiveAvgPool1d(1)

		# Replace decoder with UNet2DConditionModel
		self.decoder = UNet2DConditionModel(
			in_channels=in_channels,
			out_channels=out_channels,
			cond_dim=z_channels,
			block_out_channels=(64, 128, 256, 512),
			layers_per_block=1
		)

	def forward(self, x1, x2):
		z_diff = self.encoder(x2 - x1)  # [B, z_channels, H', W']
		
		z_diff = z_diff.view(z_diff.size(0), z_diff.size(1), -1)  # [B, z_channels, N]
		z_diff = self.global_pool(z_diff).squeeze(-1)  # [B, z_channels]

		# Decode using UNet2D
		recon = self.decoder(
			sample=x1,
			cond=z_diff,
			return_dict=False
		)[0]  # UNet returns a tuple, we want the sample
		return recon