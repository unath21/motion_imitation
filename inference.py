# -*- coding: utf-8 -*-
import os
import time
import torch
from tqdm import tqdm
import argparse
import wandb

from models import create_model, Encoder, Decoder
from utils import setup_seed, load_config
from dataset import SimplePairDataset
from accelerate import Accelerator, DataLoaderConfiguration


def initialize_model(config, accelerator):
	"""Initialize model and optional VAE encoder/decoder."""
	model_type = config["model"]["type"]
	latent_dim = config["model"]["latent_dim"]

	if "latent_inputs" in model_type:
		vae_encoder = Encoder.from_pretrained(
			"stabilityai/stable-diffusion-2-1",
			subfolder="vae",
			torch_dtype=torch.bfloat16,
		).to(accelerator.device).eval()
		vae_decoder = Decoder.from_pretrained(
			"stabilityai/stable-diffusion-2-1",
			subfolder="vae",
			torch_dtype=torch.bfloat16,
		).to(accelerator.device).eval()
		model = create_model(
			name=model_type,
			in_channels=vae_encoder.config.latent_channels,
			out_channels=vae_decoder.config.latent_channels,
			z_channels=latent_dim,
			encoder_block_out_channels=(64, 128, 256),
			decoder_cond_flatten=True if model_type == "latent_inputs_v2" else False,
			decoder_cond_scale=64 if model_type == "latent_inputs_v2" else 1,
		)
		return model, vae_encoder, vae_decoder
	model = create_model(
			name=model_type,
			in_channels=3,
			out_channels=3,
			z_channels=latent_dim,
			encoder_block_out_channels=(64, 128, 256),
			decoder_cond_flatten=True if 'v2' in model_type else False,
			decoder_cond_scale=64 * 64 if 'v2' in model_type else 1,
		)
	return model, None, None


def denormalize(tensor, config):
	"""Undo dataset normalization for visualization."""
	if "latent_inputs" in config["model"]["type"]:
		mean = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(1, 3, 1, 1)
		std = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(1, 3, 1, 1)
	else:
		mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
		std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
	return torch.clamp(tensor * std + mean, 0, 1)


@torch.inference_mode()
def upload_visualizations(model, vae_encoder, vae_decoder, batch, config, accelerator, wandb_run, split_name, epoch):
	"""Exact same W&B image logging logic as in training."""
	device = accelerator.device
	num_samples = config["validation"]["num_samples_to_log"]

	img1 = batch["img1"][:num_samples].to(device)
	img2 = batch["img2"][:num_samples].to(device)
	delta = batch["delta"][:num_samples]

	# optional fields
	dino1 = batch.get("dino1")
	dino2 = batch.get("dino2")
	masked_img1 = batch.get("masked_img1")
	masked_img2 = batch.get("masked_img2")

	model.eval()
	with torch.no_grad():
		with accelerator.autocast():
			if config["model"]["type"] == "dino" and dino1 is not None:
				predicted_img2, _ = model(img1, img2, dino1, dino2)
			elif config["model"]["type"] in ["masked_inputs", "masked_inputs_v2"] and masked_img1 is not None:
				predicted_img2, _ = model(img1, img2, masked_img1)
			elif "latent_inputs" in config["model"]["type"]:
				x1 = vae_encoder(img1).latent_dist.sample()
				x2 = vae_encoder(img2).latent_dist.sample()
				x2_pred, _ = model(x1, x2)
				predicted_img2 = vae_decoder(x2_pred, return_dict=False)[0]
			else:
				predicted_img2, _ = model(img1, img2)
	model.train()

	# --- denormalize ---
	img1_vis = denormalize(img1, config)
	img2_vis = denormalize(img2, config)
	pred_vis = denormalize(predicted_img2, config)
	if masked_img1 is not None:
		masked_img1_vis = denormalize(masked_img1, config)
		masked_img2_vis = denormalize(masked_img2, config)

	# --- make W&B image lists ---
	rows_inputs, rows_preds, rows_masked = [], [], []
	B = img1_vis.shape[0]

	for i in range(B):
		if masked_img1 is not None:
			masked_row = torch.cat([masked_img1_vis[i], pred_vis[i], masked_img2_vis[i]], dim=2)
			masked_np = (masked_row.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
			rows_masked.append(wandb.Image(masked_np, caption=f"Epoch {epoch} • {split_name} sample {i}: Masked I1 | Pred | I2"))

		input_row = torch.cat([img1_vis[i].cpu(), pred_vis[i].cpu(), img2_vis[i].cpu()], dim=2)
		input_np = (input_row.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
		rows_inputs.append(
			wandb.Image(input_np, caption=f"Epoch {epoch} • {split_name} sample {i}: Input | Pred | Target • Δ: {delta[i].cpu().numpy()}")
		)

		pred_row = torch.cat([pred_vis[i].cpu()], dim=2)
		pred_np = (pred_row.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
		rows_preds.append(wandb.Image(pred_np, caption=f"Epoch {epoch} • {split_name} sample {i}: Predicted I2"))

	wandb_log_dict = {
		f"{split_name}/comparisons_list/inputs": rows_inputs,
		f"{split_name}/comparisons_list/predictions": rows_preds,
	}
	if masked_img1 is not None:
		wandb_log_dict[f"{split_name}/comparisons_list/masked_inputs"] = rows_masked

	wandb_run.log(wandb_log_dict, step=epoch)
	print(f"[W&B] Uploaded {split_name} visualizations for checkpoint epoch {epoch}")


def main():
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
	parser.add_argument('--checkpoint_dir', type=str, required=True)
	parser.add_argument('--dino_correspondence', action='store_true', default=False, help='Enable dual guidance with DINO features and image subtraction')
	parser.add_argument('--model', type=str, choices=['autoencoder', 'simplevit', 'dino', 'masked_inputs', 'masked_inputs_v2', 'latent_inputs', 'latent_inputs_v2', 'autoencoder_v2'], required=True, help='Model type to use')
	parser.add_argument('--wandb_run_name', type=str, help='WandB run name')
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
	if args.model is not None:
		config['model']['type'] = args.model
	if args.wandb_run_name is not None:
		config['wandb']['run_name'] = args.wandb_run_name
	if args.latent_dim is not None:
		config['model']['latent_dim'] = args.latent_dim

	print(f"Loaded config from: {args.config}")

	setup_seed(config['seed'])

	dataloader_config = DataLoaderConfiguration(split_batches=False)
	accelerator = Accelerator(
		mixed_precision=config["accelerate"]["mixed_precision"],
		dataloader_config=dataloader_config,
	)

	# --- datasets ---
	train_dataset = SimplePairDataset(
		root=config["data"]["root"],
		pairs_path=config["data"]["train_pairs_json"],
		model_type=config["model"]["type"],
		frame_size=config["data"]["frame_size"],
		split="train",
		normalization="default" if "latent_inputs" in config["model"]["type"] else "imagenet",
	)
	val_dataset = SimplePairDataset(
		root=config["data"]["root"],
		pairs_path=config["data"]["val_pairs_json"],
		model_type=config["model"]["type"],
		frame_size=config["data"]["frame_size"],
		split="val",
		normalization="default" if "latent_inputs" in config["model"]["type"] else "imagenet",
	)

	# fixed batches (exact same as training visualization)
	fixed_train_batch = next(iter(torch.utils.data.DataLoader(
		train_dataset,
		batch_size=config["validation"]["num_samples_to_log"],
		shuffle=False,
		num_workers=config["data"]["num_workers"],
		pin_memory=True,
		persistent_workers=True
	)))
	fixed_val_batch = next(iter(torch.utils.data.DataLoader(
		val_dataset,
		batch_size=config["validation"]["num_samples_to_log"],
		shuffle=False,
		num_workers=config["data"]["num_workers"],
		pin_memory=True,
		persistent_workers=True
	)))

	# --- initialize model & wandb ---
	model, vae_encoder, vae_decoder = initialize_model(config, accelerator)
	model.to(accelerator.device)
	if vae_encoder:
		vae_encoder.to(accelerator.device)
		vae_decoder.to(accelerator.device)

	wandb_run = wandb.init(
		project=config['wandb']['project'],
		name=args.wandb_run_name,
		config=config,
		mode=config["wandb"].get("mode", "online"),
		dir='/scratch/rgoel15/wandb_logs'
	)

	def extract_epoch_num(filename: str) -> int:
		"""
		Extracts numeric epoch value from checkpoint filenames like:
		'checkpoint_epoch_50.pt' → 50
		If no epoch is found, returns -1.
		"""
		try:
			name = os.path.basename(filename)
			if "epoch" in name:
				return int(name.split("epoch_")[-1].split(".")[0])
			else:
				return -1
		except Exception:
			return -1

	# Sort by epoch number, not lexicographic filename order
	checkpoints = sorted(
		[f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt")],
		key=extract_epoch_num
	)

	print(f"Found {len(checkpoints)} checkpoints.")

	for ckpt_name in checkpoints:
		ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
		epoch = int(ckpt_name.split("_")[-1].split(".")[0]) if "epoch" in ckpt_name else 0
		print(f"\n[Checkpoint] Loading {ckpt_name} (epoch {epoch})")
		print(ckpt_path)
		state = torch.load(ckpt_path, map_location="cpu")
		model.load_state_dict(state["model_state_dict"], strict=False)
		model.eval()

		# --- log train and val batches just like training ---
		upload_visualizations(model, vae_encoder, vae_decoder, fixed_train_batch, config, accelerator, wandb_run, "train", epoch)
		upload_visualizations(model, vae_encoder, vae_decoder, fixed_val_batch, config, accelerator, wandb_run, "validation", epoch)

	wandb_run.finish()
	print("\n✅ All visualizations uploaded to W&B.")


if __name__ == "__main__":
	main()
