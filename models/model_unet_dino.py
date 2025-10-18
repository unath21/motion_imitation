import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.models.autoencoders.vae import Encoder

dinov2_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitl14_reg")

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


class DownBlock2D(nn.Module):
	def __init__(self, in_channels, out_channels, num_layers=2, cond_dim=None):
		super().__init__()

		layers = []
		layers.append(ResNetBlock2D(in_channels, out_channels, cond_dim, stride=2))

		for _ in range(num_layers - 1):
			layers.append(ResNetBlock2D(out_channels, out_channels, cond_dim))

		self.layers = nn.ModuleList(layers)

	def forward(self, x, cond=None):
		for layer in self.layers:
			x = layer(x, cond)
		return x


class UpBlock2D(nn.Module):
	def __init__(self, in_channels, out_channels, num_layers=2, cond_dim=None, skip_channels=None):
		super().__init__()

		self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

		if skip_channels is None:
			skip_channels = out_channels

		layers = []
		first_in = out_channels + skip_channels
		layers.append(ResNetBlock2D(first_in, out_channels, cond_dim))

		for _ in range(num_layers - 1):
			layers.append(ResNetBlock2D(out_channels, out_channels, cond_dim))

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
				DownBlock2D(in_ch, out_ch, layers_per_block, cond_dim)
			)
			in_ch = out_ch

		self.up_blocks = nn.ModuleList()
		reversed_channels = list(reversed(block_out_channels))
		in_ch = block_out_channels[-1]
		for i, out_ch in enumerate(reversed_channels):
			self.up_blocks.append(
				UpBlock2D(in_ch, out_ch, layers_per_block, cond_dim, skip_channels=out_ch)
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


def get_dino_feature_correspondence(x1, x2):
	bs, d, p_s, p_s = x1.shape
	device = x1.device

	x1 = x1.permute(0, 2, 3, 1).reshape(bs, p_s * p_s, d)  # [B, N, D]
	x2 = x2.permute(0, 2, 3, 1).reshape(bs, p_s * p_s, d)  # [B, N, D]

	# Normalize features
	x1 = F.normalize(x1, p=2, dim=-1)
	x2 = F.normalize(x2, p=2, dim=-1)

	# Compute cosine similarity
	sim = torch.bmm(x1, x2.transpose(1, 2))  # [B, N, N]
	best_match_idx = sim.argmax(dim=-1)  # [B, N]

	# Create coordinate grids
	y_coords, x_coords = torch.meshgrid(torch.arange(p_s, device=device),
									  torch.arange(p_s, device=device),
									  indexing='ij')
	coords_left = torch.stack([x_coords, y_coords], dim=-1).view(1, -1, 2).expand(bs, -1, -1)  # [B, N, 2]
	coords_right = torch.stack([x_coords, y_coords], dim=-1).view(1, -1, 2).expand(bs, -1, -1)  # [B, N, 2]

	# Fix: Use proper advanced indexing for batched operation
	batch_indices = torch.arange(bs, device=device).view(bs, 1).expand(-1, p_s * p_s)  # [B, N]
	coords_right_matched = coords_right[batch_indices, best_match_idx]  # [B, N, 2]

	flow_field_patched = coords_right_matched - coords_left  # [B, N, 2]
	flow_field_pixel = flow_field_patched * 16
	flow_field_pixel = F.normalize(flow_field_pixel.float(), p=2, dim=1)  # [B, N, 2]
	flow_grid = flow_field_pixel.view(bs, p_s, p_s, 2).permute(0, 3, 1, 2)  # [B, 2, p_s, p_s]
	flow_grid = F.interpolate(flow_grid, size=(p_s*16, p_s*16), mode='bilinear', align_corners=False)

	return flow_grid

def interpolate_dino_features(x, target_size):
	x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
	return x


class AutoencoderDINO(nn.Module):
	def __init__(self, in_channels=3, out_channels=3, z_channels=128, dino_correspondence=True):
		super().__init__()
		self.encoder = Encoder(
			in_channels=9 if not dino_correspondence else in_channels + 2,
			out_channels=z_channels,
			down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
			block_out_channels=(64, 128, 256),
			layers_per_block=1,
			act_fn='silu',
			double_z=False,
			norm_num_groups=32,
			mid_block_add_attention=False,
		)

		self.global_pool = nn.AdaptiveAvgPool1d(1)

		self.decoder = UNet2DConditionModel(
			in_channels=out_channels,
			out_channels=out_channels,
			cond_dim=z_channels,
			block_out_channels=(64, 128, 256, 512),
			layers_per_block=1
		)

		self.dino_correspondence = dino_correspondence

	def forward(self, x1, x2, dino1, dino2):
		img_diff = x2 - x1
		if self.dino_correspondence:
			z_input = get_dino_feature_correspondence(dino1, dino2)
			z_input = torch.cat([img_diff, z_input], dim=1)  # [B, 3+2, H, W]
		else:
			dino1 = interpolate_dino_features(dino1, x1.shape[2:])
			dino2 = interpolate_dino_features(dino2, x1.shape[2:])
			z_input = torch.cat([img_diff, dino1, dino2], dim=1)  # [B, 3+3+3, H, W]
		z_diff = self.encoder(z_input)  # [B, z_channels, H', W']
		z_diff = z_diff.view(z_diff.size(0), z_diff.size(1), -1)  # [B, z_channels, N]
		z_diff = self.global_pool(z_diff).squeeze(-1)  # [B, z_channels]

		# Decode using UNet2D
		recon = self.decoder(sample=x1, cond=z_diff, return_dict=False)[0]
		return recon
