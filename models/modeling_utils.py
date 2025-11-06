import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.models.autoencoders.vae import Encoder

class SpatialFiLM2d(nn.Module):
	"""Spatially-adaptive FiLM layer (SPADE-like) for motion-conditioned synthesis"""
	def __init__(self, channels: int, motion_map_channels: int):
		super().__init__()
		self.channels = channels
		self.motion_map_channels = motion_map_channels
		# Generate spatial gamma and beta from motion map
		self.param_generator = nn.Sequential(
			nn.Conv2d(motion_map_channels, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 2 * channels, kernel_size=3, padding=1)
		)
	
	def forward(self, x: torch.Tensor, motion_map: torch.Tensor):
		"""
		Args:
			x: appearance features (B, C, H, W)
			motion_map: motion context (B, K, H, W)
		Returns:
			Spatially modulated features (B, C, H, W)
		"""
		B, C, H, W = x.shape
		gamma_beta = self.param_generator(motion_map)  # (B, 2C, H, W)
		gamma, beta = gamma_beta.chunk(2, dim=1)
		return gamma * x + beta

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

		# Spatial FiLM conditioning (motion maps)
		self.spatial_film1 = SpatialFiLM2d(out_channels, cond_dim) if cond_dim else None
		self.spatial_film2 = SpatialFiLM2d(out_channels, cond_dim) if cond_dim else None

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
		if self.spatial_film1 is not None and cond is not None:
			h = self.spatial_film1(h, cond)

		# Second conv
		h = self.norm2(h)
		h = self.activation(h)
		h = self.conv2(h)
		if self.spatial_film2 is not None and cond is not None:
			h = self.spatial_film2(h, cond)

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
	def __init__(self, in_channels=3, out_channels=3, k_channels=64,
				 block_out_channels=(64, 128, 256), layers_per_block=2):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.k_channels = k_channels

		# Initial convolution
		self.conv_in = nn.Conv2d(in_channels, block_out_channels[0],
								 kernel_size=3, padding=1)

		# Encoder: non-conditional down blocks
		self.down_blocks = nn.ModuleList()
		in_ch = block_out_channels[0]
		for out_ch in block_out_channels:
			self.down_blocks.append(
				DownBlock2D(in_ch, out_ch, layers_per_block)
			)
			in_ch = out_ch

		# Decoder: blocks with spatial FiLM motion conditioning
		self.up_blocks = nn.ModuleList()
		reversed_channels = list(reversed(block_out_channels))
		in_ch = block_out_channels[-1]
		for out_ch in reversed_channels:
			self.up_blocks.append(
				UpBlock2DConditional(in_ch, out_ch, layers_per_block, 
									  cond_dim=1, skip_channels=out_ch)
			)
			in_ch = out_ch

		self.conv_out = nn.Sequential(
			nn.ConvTranspose2d(block_out_channels[0], block_out_channels[0], kernel_size=2, stride=2),
			nn.GroupNorm(min(32, block_out_channels[0]), block_out_channels[0]),
			nn.SiLU(),
			nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
		)
		
		# Spatial FiLM for final output layer
		self.spatial_film_final = SpatialFiLM2d(block_out_channels[0], 1)

	def forward(self, sample, z1=None, z2=None, return_dict=False):
		"""
		Args:
			sample: source image I1 (B, 3, H, W)
			z1: motion tensor (B, K, d, d)
			z2: motion tensor (B, K, H, W)
		Returns:
			Synthesized image or dict with 'sample' key
		"""

		x = self.conv_in(sample)

		# Encoder: extract appearance without conditioning
		skips = []
		for down_block in self.down_blocks:
			x = down_block(x)
			skips.append(x)

		# Decoder: synthesize with motion conditioning
		# Motion maps are generated on-the-fly at each decoder level after upsampling
		for idx, up_block in enumerate(self.up_blocks):
			skip = skips.pop() if len(skips) > 0 else None
			
			# Generate motion map at this decoder level's output size (after upsample)
			# The output size of upsampling is 2x the input size
			if z1 is not None and z2 is not None:
				if skip is not None:
					out_size = skip.shape[-1]
				else:
					# If no skip, use 2x the current feature map size
					out_size = x.shape[-1] * 2
				# Generate motion maps for just this level
				z1_scaled = F.interpolate(z1, size=out_size, mode='bilinear', align_corners=False)
				z2_scaled = F.adaptive_avg_pool2d(z2, out_size)
				element_wise_product = z1_scaled * z2_scaled
				motion_map = torch.sum(element_wise_product, dim=1, keepdim=True)
			else:
				motion_map = None
			
			x = up_block(x, skip=skip, cond=motion_map)

		# Final upsampling with motion conditioning
		x = self.conv_out[0](x)  # ConvTranspose2d upsample
		
		if z1 is not None and z2 is not None:
			# Generate motion map at final output size
			out_size = x.shape[-1]
			z1_scaled = F.interpolate(z1, size=out_size, mode='bilinear', align_corners=False)
			z2_scaled = F.adaptive_avg_pool2d(z2, out_size)
			element_wise_product = z1_scaled * z2_scaled
			motion_map = torch.sum(element_wise_product, dim=1, keepdim=True)
			
			# Apply spatial FiLM to final features
			x = self.spatial_film_final(x, motion_map)
			print(3, x.shape, motion_map.shape)
		# Rest of conv_out (norm, activation, final conv)
		x = self.conv_out[1](x)  # GroupNorm
		x = self.conv_out[2](x)  # SiLU
		x = self.conv_out[3](x)  # Conv2d

		if return_dict:
			return {"sample": x}
		return (x,)

class ResNetBlock2D(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1):
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

		# Second conv
		h = self.norm2(h)
		h = self.activation(h)
		h = self.conv2(h)

		return h + skip


class DownBlock2D(nn.Module):
	def __init__(self, in_channels, out_channels, num_layers=2):
		super().__init__()

		layers = []
		layers.append(ResNetBlock2D(in_channels, out_channels, stride=2))

		for _ in range(num_layers - 1):
			layers.append(ResNetBlock2D(out_channels, out_channels))

		self.layers = nn.ModuleList(layers)

	def forward(self, x, cond=None):
		for layer in self.layers:
			x = layer(x, cond)
		return x


class UpBlock2D(nn.Module):
	def __init__(self, in_channels, out_channels, num_layers=2, skip_channels=None):
		super().__init__()

		self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

		if skip_channels is None:
			skip_channels = out_channels

		layers = []
		first_in = out_channels + skip_channels
		layers.append(ResNetBlock2D(first_in, out_channels))

		for _ in range(num_layers - 1):
			layers.append(ResNetBlock2D(out_channels, out_channels))

		self.layers = nn.ModuleList(layers)

	def forward(self, x, skip=None):
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
			x = layer(x)

		return x


class UNet2DModel(nn.Module):
	def __init__(self, in_channels=3, out_channels=3,
				 block_out_channels=(64, 128, 256), layers_per_block=2):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		# Initial convolution
		self.conv_in = nn.Conv2d(in_channels, block_out_channels[0],
								 kernel_size=3, padding=1)

		self.down_blocks = nn.ModuleList()
		in_ch = block_out_channels[0]
		for out_ch in block_out_channels:
			self.down_blocks.append(
				DownBlock2D(in_ch, out_ch, layers_per_block)
			)
			in_ch = out_ch

		self.conv_out1 = nn.Sequential(
			nn.ConvTranspose2d(block_out_channels[-1], block_out_channels[-1], kernel_size=2, stride=2),

			nn.GroupNorm(min(32, block_out_channels[-1]), block_out_channels[-1]),
			nn.SiLU(),
			nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1)
		)

		self.up_blocks = nn.ModuleList()
		reversed_channels = list(reversed(block_out_channels))
		in_ch = block_out_channels[-1]
		for i, out_ch in enumerate(reversed_channels):
			self.up_blocks.append(
				UpBlock2D(in_ch, out_ch, layers_per_block, skip_channels=out_ch)
			)
			in_ch = out_ch

		self.conv_out2 = nn.Sequential(
			nn.ConvTranspose2d(block_out_channels[0], block_out_channels[0], kernel_size=2, stride=2),

			nn.GroupNorm(min(32, block_out_channels[0]), block_out_channels[0]),
			nn.SiLU(),
			nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
		)


	def forward(self, sample):
		x = self.conv_in(sample)

		skips = []
		for down_block in self.down_blocks:
			x = down_block(x)
			skips.append(x)

		down = self.conv_out1(x)

		for idx, up_block in enumerate(self.up_blocks):
			skip = skips.pop() if len(skips) > 0 else None
			x = up_block(x, skip=skip)

		up = self.conv_out2(x)

		return down, up
