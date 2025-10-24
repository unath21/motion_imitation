import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.models.autoencoders.vae import Encoder
from models.modeling_utils import UNet2DConditionModel

class Autoencoder(nn.Module):
	def __init__(self, in_channels=3, out_channels=3, z_channels=128):
		super().__init__()
		self.encoder = Encoder(
			in_channels=in_channels,
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
		recon = self.decoder(sample=x1, cond=z_diff, return_dict=False)[0]
		return recon


class AutoencoderDINOCorrespondence(nn.Module):
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

	def interpolate_dino_features(self, x, target_size):
		x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
		return x

	def get_dino_feature_correspondence(self, x1, x2):
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

	def forward(self, x1, x2, dino1, dino2):
		img_diff = x2 - x1
		if self.dino_correspondence:
			z_input = self.get_dino_feature_correspondence(dino1, dino2)
			z_input = torch.cat([img_diff, z_input], dim=1)  # [B, 3+2, H, W]
		else:
			dino1 = self.interpolate_dino_features(dino1, x1.shape[2:])
			dino2 = self.interpolate_dino_features(dino2, x1.shape[2:])
			z_input = torch.cat([img_diff, dino1, dino2], dim=1)  # [B, 3+3+3, H, W]
		z_diff = self.encoder(z_input)  # [B, z_channels, H', W']
		z_diff = z_diff.view(z_diff.size(0), z_diff.size(1), -1)  # [B, z_channels, N]
		z_diff = self.global_pool(z_diff).squeeze(-1)  # [B, z_channels]

		# Decode using UNet2D
		recon = self.decoder(sample=x1, cond=z_diff, return_dict=False)[0]
		return recon


class AutoencoderMaskedInputs(nn.Module):
	def __init__(self, in_channels=3, out_channels=3, z_channels=128):
		super().__init__()
		self.encoder = Encoder(
			in_channels=in_channels,
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
			in_channels=in_channels,
			out_channels=out_channels,
			cond_dim=z_channels,
			block_out_channels=(64, 128, 256, 512),
			layers_per_block=1
		)

	def forward(self, x1, x2, masked_x1):
		img_diff = x2 - x1
		z_diff = self.encoder(img_diff)  # [B, z_channels, H', W']
		z_diff = z_diff.view(z_diff.size(0), z_diff.size(1), -1)  # [B, z_channels, N]
		z_diff = self.global_pool(z_diff).squeeze(-1)  # [B, z_channels]

		# Decode using UNet2D
		recon = self.decoder(sample=masked_x1, cond=z_diff, return_dict=False)[0]
		return recon

class AutoencoderLatentInputs(nn.Module):
	def __init__(self, in_channels=3, z_channels=128):
		super().__init__()
		self.encoder = Encoder(
			in_channels=in_channels,
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
			in_channels=in_channels,
			out_channels=in_channels,
			cond_dim=z_channels,
			block_out_channels=(64, 128, 256, 512),
			layers_per_block=1
		)

	def forward(self, x1, x2):
		img_diff = x2 - x1
		z_diff = self.encoder(img_diff)                    # [B, z_channels, H', W']
		z_diff = z_diff.view(z_diff.size(0), z_diff.size(1), -1)  # [B, z_channels, N]
		z_diff = self.global_pool(z_diff).squeeze(-1)  # [B, z_channels]

		# --- Decode latent diff using conditional UNet ---
		recon_latent = self.decoder(sample=x1, cond=z_diff, return_dict=False)[0]
		return recon_latent


class AutoencoderV2(nn.Module):
	def __init__(self, in_channels=3, out_channels=3, z_channels=128):
		super().__init__()
		self.encoder = Encoder(
			in_channels=in_channels,
			out_channels=z_channels,
			down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
			block_out_channels=(64, 128, 256),
			layers_per_block=1,
			act_fn='silu',
			double_z=False,
			norm_num_groups=32,
			mid_block_add_attention=False,
		)

		self.decoder = UNet2DConditionModel(
			in_channels=in_channels,
			out_channels=out_channels,
			cond_dim=z_channels * 64 * 64,
			block_out_channels=(64, 128, 256, 512),
			layers_per_block=1
		)

	def forward(self, x1, x2):
		z_diff = self.encoder(x2 - x1)  # [B, z_channels, H', W']
		z_diff = z_diff.flatten(1, -1)  # [B, z_channels * H' * W']

		# Decode using UNet2D
		recon = self.decoder(sample=x1, cond=z_diff, return_dict=False)[0]
		return recon