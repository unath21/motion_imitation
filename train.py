import os
import argparse
import time
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import create_model
from utils import setup_seed, load_config, load_checkpoint, save_checkpoint
from simple_pair_dataset import SimplePairDataset

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


def initialize_model(config, accelerator):
	model_type = config["model"]["type"]
	latent_dim = config["model"]["latent_dim"]

	model = create_model(
			name=model_type,
			in_channels=3,
			out_channels=3,
			z_channels=latent_dim,
			encoder_block_out_channels=(64, 128, 256),
			decoder_cond_scale=64 * 64 if 'v2' in model_type else 1,
		)
	return model


def denormalize(tensor):
	mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
	std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
	return torch.clamp(tensor * std + mean, 0, 1)


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
	parser.add_argument('--resume', type=str, default=None,
					   help='Path to checkpoint to resume from')
	parser.add_argument('--save_dir', type=str, required=True)
	parser.add_argument('--model', type=str, choices=['autoencoder', 'simplevit', 'autoencoder_v2'], required=True, help='Model type to use')
	parser.add_argument('--wandb_name', type=str, help='WandB run name')
	parser.add_argument('--latent_dim', type=int, required=True, help='Latent dimension size for the model')

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
	if args.save_dir is not None:
		config['logging']['model_save_path'] = args.save_dir
		os.makedirs(args.save_dir, exist_ok=True)
	if args.model is not None:
		config['model']['type'] = args.model
	if args.wandb_name is not None:
		config['wandb']['run_name'] = args.wandb_name
	if args.latent_dim is not None:
		config['model']['latent_dim'] = args.latent_dim

	print(f"Loaded config from: {args.config}")

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
		split='train',
		normalization='default' if "latent_inputs" in config['model']['type'] else 'imagenet',
	)

	# For validation, we can use a subset or create separate pairs
	val_dataset = SimplePairDataset(
		root=config['data']['root'],
		pairs_path=config['data']['val_pairs_json'],
		frame_size=config['data']['frame_size'],
		split='val',
		normalization='default' if "latent_inputs" in config['model']['type'] else 'imagenet',
	)

	dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=config['data']['num_workers'], pin_memory=True, persistent_workers=True)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=config['data']['num_workers'], pin_memory=True, persistent_workers=True)

	# Cache fixed visualization batch once
	fixed_train_batch = next(iter(torch.utils.data.DataLoader(
		train_dataset,
		batch_size=config['validation']['num_samples_to_log'],
		shuffle=False,
		num_workers=config['data']['num_workers'],
		pin_memory=True,
		persistent_workers=True
	)))


	# Create tensorboard writer (only on main process)
	if accelerator.is_main_process:
		writer = SummaryWriter(config['logging']['tensorboard_log_dir'])
	else:
		writer = None

	model = initialize_model(config, accelerator)

	model = torch.compile(model, mode='reduce-overhead', fullgraph=False)

	if accelerator.is_main_process:
		print(str(model))

	# Adjust learning rate for effective batch size
	base_lr = config['train']['base_learning_rate'] * effective_batch_size / 256

	optim = torch.optim.AdamW(
		model.parameters(),
		lr=base_lr,
		betas=(0.9, 0.95),
		weight_decay=config['train']['weight_decay']
	)

	model, optim, dataloader, val_dataloader = accelerator.prepare(
		model, optim, dataloader, val_dataloader
	)

	# Load checkpoint if resuming
	start_epoch = 0
	checkpoint_path = args.resume if args.resume else None
	if checkpoint_path:
		start_epoch, _ = load_checkpoint(accelerator, model, optim, checkpoint_path)

	if accelerator.is_main_process:
		total_params = sum(p.numel() for p in model.parameters())
		trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Model - Total params: {total_params/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M")

	step_count = 0
	for e in range(start_epoch, config['total_epoch']):
		model.train()
		losses = []

		for batch in tqdm(iter(dataloader), disable=not accelerator.is_main_process):
			step_count += 1
			img1 = batch['img1']  # First image
			img2 = batch['img2']  # Second image (target)

			# Use accelerate's autocast and gradient accumulation
			with accelerator.autocast():
				predicted_img2, _ = model(img1, img2)
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

		# Calculate average training loss
		avg_train_loss = sum(losses) / len(losses)

		# Validation loss computation
		model.eval()
		val_losses = []
		with torch.no_grad():
			for val_batch in tqdm(iter(val_dataloader), disable=not accelerator.is_main_process, desc="Validation"):
				val_img1 = val_batch['img1']
				val_img2 = val_batch['img2']

				with accelerator.autocast():
					predicted_val_img2, _ = model(val_img1, val_img2)
					val_loss = torch.mean((predicted_val_img2 - val_img2) ** 2)

				# Gather validation loss for logging
				val_loss_gathered = accelerator.gather_for_metrics(val_loss)
				val_losses.append(val_loss_gathered.mean().item())

		# Calculate average validation loss
		avg_val_loss = sum(val_losses) / len(val_losses)

		# Log and print (only from main process)
		if accelerator.is_main_process:
			if writer is not None:
				writer.add_scalar('train/loss', avg_train_loss, global_step=e)
				writer.add_scalar('validation/loss', avg_val_loss, global_step=e)
				current_lr = base_lr
				writer.add_scalar('learning_rate', current_lr, global_step=e)

			# WandB logging
			if wandb_run is not None:
				wandb_log_dict = {
					'train/loss': avg_train_loss,
					'validation/loss': avg_val_loss,
					'train/learning_rate': current_lr,
					'epoch': e
				}

				# Log training images periodically
				if e % config['validation']['log_images_every'] == 0:
					# Get a batch for training visualization
					train_batch = fixed_train_batch
					train_img1 = train_batch['img1'][:config['validation']['num_samples_to_log']]
					train_img2 = train_batch['img2'][:config['validation']['num_samples_to_log']]
					train_delta = train_batch['delta'][:config['validation']['num_samples_to_log']]

					model.eval()
					with torch.no_grad():
						with accelerator.autocast():
							predicted_train_img2, _ = model(train_img1, train_img2)
					model.train()

					train_img1_vis = denormalize(train_img1)
					train_img2_vis = denormalize(train_img2)
					predicted_train_vis = denormalize(predicted_train_img2)

					# Create training image rows for WandB
					train_rows = []
					train_masked_rows = []
					train_pred_rows = []
					B = train_img1_vis.shape[0]
					for i in range(B):
						row = torch.cat([train_img1_vis[i].cpu(), predicted_train_vis[i].cpu(), train_img2_vis[i].cpu()], dim=2)
						row_np = (row.permute(1, 2, 0).numpy() * 255).astype('uint8')
						train_rows.append(wandb.Image(row_np, caption=f"Epoch {e} • train sample {i}: Input | Pred | Target • Δ: {train_delta[i].cpu().numpy()}"))

						pred_row = torch.cat([predicted_train_vis[i].cpu()], dim=2)
						pred_row_np = (pred_row.permute(1, 2, 0).numpy() * 255).astype('uint8')
						train_pred_rows.append(wandb.Image(pred_row_np, caption=f"Epoch {e} • train sample {i}: Predicted I2"))

					wandb_log_dict['train/comparisons_list/inputs'] = train_rows
					wandb_log_dict['train/comparisons_list/predictions'] = train_pred_rows

				wandb_run.log(wandb_log_dict, step=e)

			print(f'Epoch {e}, avg train loss: {avg_train_loss:.6f}, avg val loss: {avg_val_loss:.6f}, lr: {current_lr:.2e}')

		# Validation visualization (every N epochs)
		if e % config['validation']['log_images_every'] == 0:
			model.eval()
			with torch.no_grad():
				val_batch = next(iter(val_dataloader))
				val_img1 = val_batch['img1'][:config['validation']['num_samples_to_log']]
				val_img2 = val_batch['img2'][:config['validation']['num_samples_to_log']]
				val_delta = val_batch['delta'][:config['validation']['num_samples_to_log']]

				with accelerator.autocast():
					predicted_val_img2, _ = model(val_img1, val_img2)

				# Only visualize from main process
				if accelerator.is_main_process and writer is not None:
					# Denormalize for visualization
					val_img1_vis = denormalize(val_img1)
					val_img2_vis = denormalize(val_img2)
					predicted_vis = denormalize(predicted_val_img2)

					if wandb_run is not None:
						rows = []
						masked_input_rows = []
						pred_rows = []
						B = val_img1_vis.shape[0]
						for i in range(B):
							row = torch.cat([val_img1_vis[i].cpu(), predicted_vis[i].cpu(), val_img2_vis[i].cpu()], dim=2)
							row_np = (row.permute(1, 2, 0).numpy() * 255).astype('uint8')
							rows.append(wandb.Image(row_np, caption=f"Epoch {e} • sample {i}: Input | Pred | Target • Δ: {val_delta[i].cpu().numpy()}"))

							pred_row = torch.cat([predicted_vis[i].cpu()], dim=2)
							pred_row_np = (pred_row.permute(1, 2, 0).numpy() * 255).astype('uint8')
							pred_rows.append(wandb.Image(pred_row_np, caption=f"Epoch {e} • val sample {i}: Predicted I2"))

						wandb_run.log({'validation/comparisons_list/inputs': rows}, step=e)
						wandb_run.log({'validation/comparisons_list/predictions': pred_rows}, step=e)

		# Save checkpoint (only from main process) - Updated to match your original style
		if accelerator.is_main_process and (e % config['validation']['save_model_every'] == 0 or e == config['total_epoch'] - 1):
			checkpoint_dir = config['logging']['model_save_path']
			checkpoint_filename = f"checkpoint_epoch_{e}.pt"
			checkpoint_save_path = os.path.join(checkpoint_dir, checkpoint_filename)
			save_checkpoint(accelerator, model, optim, e, avg_train_loss, checkpoint_save_path)

	if accelerator.is_main_process:
		print("Training complete!")
		if writer is not None:
			writer.close()
		if wandb_run is not None:
			wandb_run.finish()
			print("WandB run finished.")