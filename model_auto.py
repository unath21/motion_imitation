import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from diffusers import UNet2DConditionModel
from diffusers.models.autoencoders.vae import Encoder

# Swish/SiLU activation function
def swish(x):
	return x * torch.sigmoid(x)

class EssenceExtractor(nn.Module):
	def __init__(self, embed_dim=256, latent_dim=64):
		super().__init__()
		# Use adaptive pooling to handle variable spatial dimensions
		self.global_pool = nn.AdaptiveAvgPool1d(1)  # Pool to 1x1 spatial
		self.extractor = nn.Sequential(
			nn.Linear(in_features=embed_dim, out_features=embed_dim // 2),
			nn.ReLU(),
			nn.Linear(in_features=embed_dim // 2, out_features=latent_dim),
		)
		self.init_weight()

	def init_weight(self):
		for m in self.extractor:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				nn.init.zeros_(m.bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		B, C, H, W = x.shape
		x = x.view(B, C, H * W)  # (B, C, H*W)
		x = self.global_pool(x).squeeze(-1)  # (B, C)
		x = self.extractor(x)  # (B, latent_dim)
		return x

class Autoencoder(nn.Module):
	def __init__(self, in_channels=3, out_channels=3, z_channels=4, sample_size=64, essence_dim=64):
		super().__init__()
		self.encoder = Encoder(
			in_channels=in_channels,
			out_channels=256,
			down_block_types=("DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"),
			block_out_channels=(64, 128, 256),
			layers_per_block=2,
			act_fn='silu',
			double_z=False,
			norm_num_groups=32,
			mid_block_add_attention=True,
		)

		self.essence_extractor = EssenceExtractor(
			embed_dim=256,
			latent_dim=essence_dim
		)

		# Replace decoder with UNet2DConditionModel
		self.decoder = UNet2DConditionModel(
			sample_size=sample_size // 2,  # Size after encoding
			in_channels=in_channels,
			out_channels=out_channels,
			layers_per_block=2,
			block_out_channels=(64, 128, 256),
			down_block_types=(
				"CrossAttnDownBlock2D",
				"CrossAttnDownBlock2D",
				"DownBlock2D",
			),
			up_block_types=(
				"UpBlock2D",
				"CrossAttnUpBlock2D",
				"CrossAttnUpBlock2D",
			),
			cross_attention_dim=essence_dim,  # Should match essence_extractor output
			attention_head_dim=8,
		)

	def forward(self, x1, x2):
		# Encode the difference between frames
		z_diff = self.encoder(x2 - x1)  # Shape: [B, z_channels, H//2, W//2]

		# Extract essence for conditioning
		essence = self.essence_extractor(z_diff)  # Shape: [B, essence_dim]

		# Generate dummy timesteps (since we're not doing diffusion)
		B = z_diff.shape[0]
		timesteps = torch.zeros(B, device=z_diff.device, dtype=torch.long)

		# Decode using UNet2D
		recon = self.decoder(
			sample=x1,
			timestep=timesteps,
			encoder_hidden_states=essence.unsqueeze(1),
			return_dict=False
		)[0]  # UNet returns a tuple, we want the sample

		return recon