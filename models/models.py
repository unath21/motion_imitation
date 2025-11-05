import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.models.autoencoders.vae import Encoder
from models.modeling_utils import UNet2DConditionModel


# ============================================================
# Model Registry
# ============================================================

MODEL_REGISTRY = {}

def register_model(name: str):
    """Decorator to register a model in the global MODEL_REGISTRY."""
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered.")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str):
    """Return model class from registry."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]


def create_model(name: str, **kwargs):
    """Instantiate a registered model by name."""
    cls = get_model_class(name)
    return cls(**kwargs)


# ============================================================
# Base class
# ============================================================

class BaseAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        z_channels=128,
        encoder_block_out_channels=(64, 128, 256),
        decoder_block_out_channels=(64, 128, 256, 512),
        decoder_cond_flatten=False,
        decoder_cond_scale=1,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=z_channels,
            down_block_types=("DownEncoderBlock2D",) * len(encoder_block_out_channels),
            block_out_channels=encoder_block_out_channels,
            layers_per_block=1,
            act_fn="silu",
            double_z=False,
            norm_num_groups=32,
            mid_block_add_attention=False,
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.decoder_cond_flatten = decoder_cond_flatten
        self.decoder_cond_scale = decoder_cond_scale

        cond_dim = z_channels * decoder_cond_scale
        self.decoder = UNet2DConditionModel(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dim=cond_dim,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=1,
        )

    def encode(self, x):
        z = self.encoder(x)
        if self.decoder_cond_flatten:
            return z.flatten(1)
        z = z.view(z.size(0), z.size(1), -1)
        return self.global_pool(z).squeeze(-1)

    def decode(self, sample, cond):
        return self.decoder(sample=sample, cond=cond, return_dict=False)[0]


# ============================================================
# Variants
# ============================================================

@register_model("autoencoder")
class Autoencoder(BaseAutoencoder):
    def forward(self, x1, x2):
        z_diff = self.encode(x2 - x1)
        recon = self.decode(x1, z_diff)
        return recon, z_diff


@register_model("autoencoder_v2")
class AutoencoderV2(BaseAutoencoder):
    def forward(self, x1, x2):
        z_diff = self.encode(x2 - x1)
        recon = self.decode(x1, z_diff)
        return recon, z_diff
