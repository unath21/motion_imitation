import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        # patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask
    

class SimpleEncoder(torch.nn.Module):
    def __init__(self, image_size=32, patch_size=2, emb_dim=192, num_layer=12, num_head=3):
        super().__init__()
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img): # [512, 3, 32, 32]
        patches = self.patchify(img) # [512, 192, 16, 16]
        patches = rearrange(patches, 'b c h w -> (h w) b c') # [256, 512, 192]
        patches = patches + self.pos_embedding
        patches = rearrange(patches, 't b c -> b t c') # [512, 256, 192]
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c') # [256, 512, 192]
        return features
    
class SimpleEncoderDecoder(torch.nn.Module):
    def __init__(self, image_size=32, patch_size=2, emb_dim=192, num_layer=12, num_head=3, latent_dim=64):
        super().__init__()
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))
        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.essence_projection = torch.nn.Linear(latent_dim, emb_dim)
        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, z, img): # [512, 3, 32, 32]
        z_token = self.essence_projection(z)  # (batch, emb_dim)
        z_token = rearrange(z_token, 'b c -> 1 b c')  # (1, batch, emb_dim)

        patches = self.patchify(img) # [512, 192, 16, 16]
        patches = rearrange(patches, 'b c h w -> (h w) b c') # [256, 512, 192]
        patches = torch.cat([z_token, patches], dim=0)
        
        features = patches + self.pos_embedding
        
        features = rearrange(features, 't b c -> b t c')
        features = self.layer_norm(self.transformer(features))
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # (num_patches, batch, emb_dim)
        
        # Generate patches
        patches = self.head(features)
        img = self.patch2img(patches)
        return img
    
class SimpleDecoder(torch.nn.Module):
    def __init__(self, image_size=32, patch_size=2, emb_dim=192, num_layer=4, num_head=3, latent_dim=64):
        super().__init__()
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)
        
        self.essence_projection = torch.nn.Linear(latent_dim, emb_dim)
        
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.pos_embedding, std=.02)
        trunc_normal_(self.essence_projection.weight, std=.02)
        if self.essence_projection.bias is not None:
            torch.nn.init.constant_(self.essence_projection.bias, 0)

    def forward(self, z, features):
        # z: (batch, latent_dim) - essence vector
        z_token = self.essence_projection(z)  # (batch, emb_dim)
        z_token = rearrange(z_token, 'b c -> 1 b c')  # (1, batch, emb_dim)
        
        # Concatenate essence token with features
        features = torch.cat([z_token, features], dim=0)  # (num_patches + 1, batch, emb_dim)
        
        # Add positional embedding
        features = features + self.pos_embedding
        
        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # (num_patches, batch, emb_dim)
        
        # Generate patches
        patches = self.head(features)
        img = self.patch2img(patches)
        return img

class SimpleEssenceExtractor(torch.nn.Module):
    def __init__(self, emb_dim=192, latent_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        
        # Global average pooling followed by projection
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim // 2, latent_dim),
        )
        
        # Initialize weights
        self.init_weight()
    
    def init_weight(self):
        for m in self.projection:
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (num_patches, batch, emb_dim)
        x = rearrange(x, 't b c -> b c t')  # (batch, emb_dim, num_patches)
        x = self.global_pool(x)  # (batch, emb_dim, 1)
        x = x.squeeze(-1)  # (batch, emb_dim)
        essence = self.projection(x)  # (batch, latent_dim)
        return essence
    
class SimpleViT(torch.nn.Module):
    def __init__(self, image_size=32, patch_size=2, emb_dim=192, encoder_layer=12, encoder_head=3, decoder_layer=4, decoder_head=3, latent_dim=64):
        super().__init__()
        self.encoder = SimpleEncoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head)
        self.essence_extractor = SimpleEssenceExtractor(emb_dim, latent_dim)
        self.decoder = SimpleEncoderDecoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head, latent_dim)

    def forward(self, img1, img2):
        x1 = self.encoder(img2 - img1)  # (num_patches, batch, emb_dim)
        # x2 = self.encoder(img2)  # (num_patches, batch, emb_dim)
        z = self.essence_extractor(x1)  # (batch, latent_dim)
        img2_pred = self.decoder(z, img1)  # (batch, 3, height, width)

        return img2_pred


if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(img)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)
