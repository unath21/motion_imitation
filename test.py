import torch
from accelerate import Accelerator
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# ==== USER IMPORTS (edit these for your project) ====
from models import *    # e.g. from models.model_auto import AutoUNet
from simple_pair_dino_dataset import SimplePairDINODataset   # e.g. from dataset import ValidationDataset
# ====================================================

# Paths and config
checkpoint_path = "/scratch/rgoel15/motion_emitation_data/model_mimic/checkpoint_epoch_950.pt"
project_name = "motion_mimic_validation"
batch_size = 4

# Initialize accelerator and wandb
accelerator = Accelerator()
wandb.init(project=project_name, name="val_epoch_950", mode="online")

# ----- Load model -----
model = AutoencoderDINO(
            in_channels=9,
            out_channels=3,
            z_channels=64
        )

checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    # fallback if full model was saved
    model = checkpoint

model = accelerator.prepare(model)
model.eval()

# ----- Prepare validation data -----
val_dataset = SimplePairDINODataset(
            root="/data/pturaga/datasets/Penn_Action",
            pairs_path="/home/rgoel15/motion_imitation/data/val_bench.json",
            frame_size=256,
            split='val'  # Different split for different augmentations
        )
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_loader = accelerator.prepare(val_loader)

# Denormalize function for visualization (same as train.py)
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

def visualize_latent_vector(z_vector, sample_idx=0, num_channels=16):
    """Visualize latent vector as feature maps
    
    Args:
        z_vector: tensor of shape [B, 64, 64, 64]
        sample_idx: which sample from batch to visualize
        num_channels: number of channels to visualize
    """
    # Take one sample from batch
    z_sample = z_vector[sample_idx].cpu().numpy()  # [64, 64, 64]
    
    # Create subplot grid for feature channels
    cols = 4
    rows = (num_channels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_channels):
        row = i // cols
        col = i % cols
        
        if i < z_sample.shape[0]:
            channel = z_sample[i]  # [64, 64]
            
            # Normalize for visualization
            vmin, vmax = channel.min(), channel.max()
            if vmax > vmin:
                channel_norm = (channel - vmin) / (vmax - vmin)
            else:
                channel_norm = channel
            
            axes[row, col].imshow(channel_norm, cmap='viridis')
            axes[row, col].set_title(f'Channel {i}')
            axes[row, col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Convert to WandB image
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return wandb.Image(buf, caption=f"Latent features sample {sample_idx}")

def visualize_latent_statistics(z_vector):
    """Visualize statistics of the latent vector"""
    # Calculate statistics across spatial dimensions
    z_mean = z_vector.mean(dim=[2, 3])  # [B, 64]
    z_std = z_vector.std(dim=[2, 3])    # [B, 64]
    z_min = z_vector.min(dim=3)[0].min(dim=2)[0]  # [B, 64]
    z_max = z_vector.max(dim=3)[0].max(dim=2)[0]  # [B, 64]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot mean across channels
    axes[0, 0].plot(z_mean[0].cpu().numpy())
    axes[0, 0].set_title('Mean activation per channel')
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Mean value')
    
    # Plot std across channels
    axes[0, 1].plot(z_std[0].cpu().numpy())
    axes[0, 1].set_title('Std deviation per channel')
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Std value')
    
    # Plot min/max range
    axes[1, 0].fill_between(range(64), z_min[0].cpu().numpy(), z_max[0].cpu().numpy(), alpha=0.3)
    axes[1, 0].plot(z_min[0].cpu().numpy(), label='Min')
    axes[1, 0].plot(z_max[0].cpu().numpy(), label='Max')
    axes[1, 0].set_title('Min/Max range per channel')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    
    # Plot histogram of all values
    z_flat = z_vector[0].cpu().numpy().flatten()
    axes[1, 1].hist(z_flat, bins=50, alpha=0.7)
    axes[1, 1].set_title('Distribution of latent values')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Convert to WandB image
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return wandb.Image(buf, caption="Latent vector statistics")

# ----- Validation loop -----
with torch.no_grad():
    for batch_idx, val_batch in enumerate(val_loader):
        # Get validation data
        val_img1 = val_batch['img1']
        val_img2 = val_batch['img2'] 
        val_delta = val_batch['delta']
        val_dino1 = val_batch['dino1']
        val_dino2 = val_batch['dino2']
        
        # Forward pass
        output, z_vector = model(val_img1, val_img2, val_dino1, val_dino2)
        
        print(f"Batch {batch_idx}: z_vector shape = {z_vector.shape}")
        
        # Denormalize for visualization (same as train.py)
        val_img1_vis = denormalize(val_img1)
        val_img2_vis = denormalize(val_img2)
        predicted_vis = denormalize(output)
        
        # Create image rows for WandB (same format as train.py)
        rows = []
        B = val_img1_vis.shape[0]
        for i in range(B):
            row = torch.cat([val_img1_vis[i], predicted_vis[i], val_img2_vis[i]], dim=2)
            row_np = (row.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            rows.append(wandb.Image(row_np, caption=f"Batch {batch_idx} • sample {i}: Input | Pred | Target • Δ: {val_delta[i].cpu().numpy()}"))
        
        # Visualize latent vectors for first few samples
        latent_visualizations = []
        for i in range(B):  # Visualize first 2 samples per batch
            latent_viz = visualize_latent_vector(z_vector, sample_idx=i, num_channels=16)
            latent_visualizations.append(latent_viz)
        
        # Visualize latent statistics
        latent_stats = visualize_latent_statistics(z_vector)
        
        # Log to wandb (same as train.py)
        wandb_log_dict = {
            'validation/comparisons_list': rows,
            'validation/latent_features': latent_visualizations,
        }
        
        wandb.log(wandb_log_dict, step=batch_idx)
        
        # Limit uploads if large dataset
        if batch_idx >= 10:
            break

wandb.finish()
print("✅ Validation complete and images uploaded to wandb.")