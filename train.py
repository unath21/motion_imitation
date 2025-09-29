import os
import argparse
import math
import time
import yaml
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F

from model_mimic import SimpleViT
from model_dcae import DCAE, dc_ae_f32c32, dc_ae_f64c128, dc_ae_f128c512
from utils import setup_seed
from simple_pair_dataset import SimplePairDataset


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

try:
    from accelerate import Accelerator
    from accelerate import DataLoaderConfiguration
    accelerate_available = True
except ImportError:
    accelerate_available = False

try:
    import wandb
except ImportError:
    wandb = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimpleViT MAE Pretraining')
    parser.add_argument('--config', type=str, default='config_mae_pretrain.yaml',
                       help='Path to config file')
    
    # Optional overrides for key parameters
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--batch_size', type=int, help='Total batch size') 
    parser.add_argument('--max_device_batch_size', type=int, help='Per-device batch size')
    parser.add_argument('--base_learning_rate', type=float, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--total_epoch', type=int, help='Total epochs')
    parser.add_argument('--warmup_epoch', type=int, help='Warmup epochs')
    parser.add_argument('--data_root', type=str, help='Dataset root directory')
    parser.add_argument('--frame_size', type=int, help='Frame size')
    parser.add_argument('--mixed_precision', type=str, choices=['no', 'fp16', 'bf16'], 
                       help='Mixed precision mode')
    parser.add_argument('--wandb_mode', type=str, choices=['online', 'offline', 'disabled'], 
                       help='WandB mode')

    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.seed is not None:
        config['seed'] = args.seed
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
    if args.max_device_batch_size is not None:
        config['train']['max_device_batch_size'] = args.max_device_batch_size
    if args.base_learning_rate is not None:
        config['train']['base_learning_rate'] = args.base_learning_rate
    if args.weight_decay is not None:
        config['train']['weight_decay'] = args.weight_decay
    if args.total_epoch is not None:
        config['total_epoch'] = args.total_epoch
    if args.warmup_epoch is not None:
        config['warmup_epoch'] = args.warmup_epoch
    if args.data_root is not None:
        config['data']['root'] = args.data_root
    if args.frame_size is not None:
        config['data']['frame_size'] = args.frame_size
    if args.mixed_precision is not None:
        config['accelerate']['mixed_precision'] = args.mixed_precision
    if args.wandb_mode is not None:
        config['wandb']['mode'] = args.wandb_mode
    
    print(f"Loaded config from: {args.config}")
    # print(f"Using configuration: {yaml.dump(config, default_flow_style=False)}")

    setup_seed(config['seed'])

    # Initialize accelerator
    if os.path.exists(config['accelerate']['config_file']):
        os.environ['ACCELERATE_CONFIG_FILE'] = config['accelerate']['config_file']
        print(f"Using accelerate config: {config['accelerate']['config_file']}")
    
    # Create dataloader configuration to fix deprecation warning
    dataloader_config = DataLoaderConfiguration(split_batches=False)
    
    accelerator = Accelerator(
        mixed_precision=config['accelerate']['mixed_precision'],
        gradient_accumulation_steps=config['accelerate']['gradient_accumulation_steps'],
        dataloader_config=dataloader_config,
    )
    
    if accelerator.is_main_process:
        print(f"Accelerate initialized with {accelerator.num_processes} processes")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        print(f"Gradient accumulation steps: {config['accelerate']['gradient_accumulation_steps']}")

    # Initialize WandB (only on main process)
    wandb_run = None
    if config['wandb']['enable'] and config['wandb']['mode'] != 'disabled' and accelerator.is_main_process:
        if wandb is None:
            print("WandB logging requested but wandb package not installed. Skipping wandb logging.")
        else:
            # Create wandb directory
            os.makedirs(config['wandb']['dir'], exist_ok=True)
            
            # Add timestamp to run name for uniqueness
            timestamp = time.strftime("%m%d_%H%M", time.localtime())
            run_name_with_timestamp = f"{config['wandb']['run_name']}_{timestamp}"
            
            wandb_run = wandb.init(
                project=config['wandb']['project'],
                name=run_name_with_timestamp,
                config=config,  # Log entire config
                mode=config['wandb']['mode'],
                dir=config['wandb']['dir']
            )
            print(f"WandB initialized: {config['wandb']['project']}/{run_name_with_timestamp}")

    batch_size = config['train']['batch_size']
    load_batch_size = min(config['train']['max_device_batch_size'], batch_size)

    # For accelerate, load_batch_size is per device
    effective_batch_size = load_batch_size * accelerator.num_processes * config['accelerate']['gradient_accumulation_steps']
    if accelerator.is_main_process:
        print(f"Per-device batch size: {load_batch_size}")
        print(f"Effective batch size: {effective_batch_size}")

    # Create SimplePairDataset for training
    train_dataset = SimplePairDataset(
        root=config['data']['root'],
        pairs_path=config['data']['train_pairs_json'],
        frame_size=config['data']['frame_size'],
        split='train'
    )
    
    # For validation, we can use a subset or create separate pairs
    val_dataset = SimplePairDataset(
        root=config['data']['root'],
        pairs_path=config['data']['val_pairs_json'],
        frame_size=config['data']['frame_size'],
        split='val'  # Different split for different augmentations
    )
    
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=config['data']['num_workers'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=config['data']['num_workers'])
    
    # Create tensorboard writer (only on main process)
    if accelerator.is_main_process:
        writer = SummaryWriter(config['logging']['tensorboard_log_dir'])
    else:
        writer = None

    # Create model with config parameters
    if config['model']['type'] == 'SimpleViT':
        model = SimpleViT(
            image_size=config['model']['image_size'],
            patch_size=config['model']['patch_size'],
            latent_dim=config['model']['latent_dim']
        )
    elif config['model']['type'] == 'DCAE':
        if config['model']['dcae_latent_dim'] == 32:
            cfg = dc_ae_f32c32(name='dc-ae-f32c32-in-1.0', pretrained_path=None)
        elif config['model']['dcae_latent_dim'] == 64:
            cfg = dc_ae_f64c128(name='dc-ae-f64c128-in-1.0', pretrained_path=None)
        elif config['model']['dcae_latent_dim'] == 128:
            cfg = dc_ae_f128c512(name='dc-ae-f128c512-in-1.0', pretrained_path=None)
        model = DCAE(
            cfg=cfg
        )
    
    # Adjust learning rate for effective batch size
    base_lr = config['train']['base_learning_rate'] * effective_batch_size / 256
    
    optim = torch.optim.AdamW(
        model.parameters(), 
        lr=base_lr, 
        betas=(0.9, 0.95), 
        weight_decay=config['train']['weight_decay']
    )
    lr_func = lambda epoch: min((epoch + 1) / (config['warmup_epoch'] + 1e-8), 
                               0.5 * (math.cos(epoch / config['total_epoch'] * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=accelerator.is_main_process)

    # Prepare model, optimizer, scheduler and dataloaders with accelerate
    model, optim, lr_scheduler, dataloader, val_dataloader = accelerator.prepare(
        model, optim, lr_scheduler, dataloader, val_dataloader
    )
    
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"SimpleViT - Total params: {total_params/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M")

    step_count = 0
    for e in range(config['total_epoch']):
        model.train()
        losses = []
        
        for batch in tqdm(iter(dataloader), disable=not accelerator.is_main_process):
            step_count += 1
            img1 = batch['img1']  # First image
            img2 = batch['img2']  # Second image (target)
            
            # Use accelerate's autocast and gradient accumulation
            with accelerator.autocast():
                predicted_img2 = model(img1, img2)
                loss = torch.mean((predicted_img2 - img2) ** 2)
            
            accelerator.backward(loss)
            
            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config['train']['gradient_clip_norm'])
            
            optim.step()
            optim.zero_grad(set_to_none=True)
            
            # Gather loss for logging
            loss_gathered = accelerator.gather_for_metrics(loss)
            losses.append(loss_gathered.mean().item())
        # Step scheduler
        lr_scheduler.step()
        
        # Calculate average loss
        avg_loss = sum(losses) / len(losses)
        
        # Log and print (only from main process)
        if accelerator.is_main_process:
            if writer is not None:
                writer.add_scalar('reconstruction_loss', avg_loss, global_step=e)
                current_lr = lr_scheduler.get_last_lr()[0]
                writer.add_scalar('learning_rate', current_lr, global_step=e)
            
            # WandB logging
            if wandb_run is not None:
                wandb_log_dict = {
                    'train/loss': avg_loss,
                    'train/learning_rate': current_lr,
                    'epoch': e
                }
                
                # Log training images periodically
                if e % config['validation']['log_images_every'] == 0:
                    # Get a batch for training visualization
                    train_batch = next(iter(dataloader))
                    train_img1 = train_batch['img1'][:8]  # First 8 training samples
                    train_img2 = train_batch['img2'][:8]  # Target images
                    
                    model.eval()
                    with torch.no_grad():
                        with accelerator.autocast():
                            predicted_train_img2 = model(train_img1, train_img2)
                    model.train()
                    
                    # Denormalize for visualization (reverse ImageNet normalization)
                    def denormalize(tensor):
                        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
                        return torch.clamp(tensor * std + mean, 0, 1)
                    
                    train_img1_vis = denormalize(train_img1)
                    train_img2_vis = denormalize(train_img2)
                    predicted_train_vis = denormalize(predicted_train_img2)
                    
                    # Create training image rows for WandB
                    train_rows = []
                    B = train_img1_vis.shape[0]
                    for i in range(B):
                        # Row = [img1 | pred | target] concatenated along width
                        row = torch.cat([train_img1_vis[i], predicted_train_vis[i], train_img2_vis[i]], dim=2)  # [C,H,3W]
                        row_np = (row.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                        train_rows.append(wandb.Image(row_np, caption=f"Epoch {e} • train sample {i}: Input | Pred | Target"))
                    
                    wandb_log_dict['train/comparisons_list'] = train_rows
                
                wandb_run.log(wandb_log_dict, step=e)
            
            print(f'Epoch {e}, average training loss: {avg_loss:.6f}, lr: {current_lr:.2e}')

        # Validation and visualization (every N epochs)
        if e % config['validation']['log_images_every'] == 0:
            model.eval()
            with torch.no_grad():
                val_batch = next(iter(val_dataloader))
                val_img1 = val_batch['img1'][:config['validation']['num_samples_to_log']]
                val_img2 = val_batch['img2'][:config['validation']['num_samples_to_log']]
                
                with accelerator.autocast():
                    predicted_val_img2 = model(val_img1, val_img2)
                
                # Only visualize from main process
                if accelerator.is_main_process and writer is not None:
                    # Denormalize for visualization (reverse ImageNet normalization)
                    def denormalize(tensor):
                        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
                        return torch.clamp(tensor * std + mean, 0, 1)
                    
                    val_img1_vis = denormalize(val_img1)
                    val_img2_vis = denormalize(val_img2)
                    predicted_vis = denormalize(predicted_val_img2)
                    
                    # Create visualization: [img1, predicted_img2, actual_img2] for each sample
                    img_grid = torch.cat([val_img1_vis, predicted_vis, val_img2_vis], dim=0)
                    img_grid = rearrange(img_grid, '(n b) c h w -> c (b h) (n w)', n=3, b=config['validation']['num_samples_to_log'])
                    
                    if wandb_run is not None:
                        rows = []
                        B = val_img1_vis.shape[0]
                        for i in range(B):
                            # Row = [img1 | pred | target] concatenated along width
                            row = torch.cat([val_img1_vis[i], predicted_vis[i], val_img2_vis[i]], dim=2)  # [C,H,3W]
                            row_np = (row.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                            rows.append(wandb.Image(row_np, caption=f"Epoch {e} • sample {i}: Input | Pred | Target "))

                        wandb_run.log({'validation/comparisons_list': rows}, step=e)
                    
                    # # Log to WandB
                    # if wandb_run is not None:
                    #     # Convert to numpy for wandb
                    #     img_grid_np = (img_grid.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                    #     wandb_run.log({
                    #         'validation/comparison': wandb.Image(
                    #             img_grid_np, 
                    #             caption=f"Epoch {e}: Input | Predicted | Target"
                    #         )
                    #     }, step=e)

                    
        
        # Save model (only from main process) - every N epochs or at the end
        if accelerator.is_main_process and (e % config['validation']['save_model_every'] == 0 or e == config['total_epoch'] - 1):
            # Get unwrapped model for saving
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = config['logging']['model_save_path']
            if e != config['total_epoch'] - 1:
                # Add epoch number for intermediate saves
                base_name, ext = os.path.splitext(save_path)
                save_path = f"{base_name}_epoch_{e}{ext}"
            torch.save(unwrapped_model, save_path)
            print(f"Model saved to: {save_path}")

    if accelerator.is_main_process:
        print("Training complete!")
        if writer is not None:
            writer.close()
        if wandb_run is not None:
            wandb_run.finish()
            print("WandB run finished.")