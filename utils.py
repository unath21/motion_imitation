import random
import torch
import numpy as np
import os
import yaml

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path):
	"""Load configuration from YAML file."""
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config


def save_checkpoint(accelerator, model, optimizer, epoch, loss, checkpoint_path):
	"""Save training checkpoint, compatible with torch.compile() and Accelerate."""
	if accelerator.is_main_process:
		# Unwrap the model (Accelerate handles DDP)
		unwrapped_model = accelerator.unwrap_model(model)

		# If the model is compiled, get the original module
		if hasattr(unwrapped_model, '_orig_mod'):
			unwrapped_model = unwrapped_model._orig_mod

		clean_state_dict = unwrapped_model.state_dict()

		checkpoint = {
			'epoch': epoch,
			'model_state_dict': clean_state_dict,  # Save clean state dict
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss,
		}

		torch.save(checkpoint, checkpoint_path)
		print(f"✅ Checkpoint saved (epoch {epoch}) at: {checkpoint_path}")


def load_checkpoint(accelerator, model, optimizer, checkpoint_path):
	"""Load training checkpoint with compatibility for torch.compile() and Accelerate."""
	if not os.path.exists(checkpoint_path):
		print(f"⚠️ No checkpoint found at: {checkpoint_path}")
		return 0, None

	print(f"📂 Loading checkpoint from: {checkpoint_path}")
	checkpoint = torch.load(checkpoint_path, map_location='cpu')

	# Unwrap model for loading
	unwrapped_model = accelerator.unwrap_model(model)
	is_compiled_model = hasattr(unwrapped_model, '_orig_mod')

	model_to_load = unwrapped_model._orig_mod if is_compiled_model else unwrapped_model
	checkpoint_state_dict = checkpoint.get('model_state_dict', checkpoint)

	# Handle potential prefix mismatch for compiled models
	if is_compiled_model:
		model_keys = model_to_load.state_dict().keys()
		has_prefix_in_model = any(k.startswith('_orig_mod.') for k in model_keys)
		has_prefix_in_ckpt = any(k.startswith('_orig_mod.') for k in checkpoint_state_dict)

		if has_prefix_in_model and not has_prefix_in_ckpt:
			print("ℹ️ Adding '_orig_mod.' prefix to checkpoint keys for compiled model.")
			checkpoint_state_dict = {'_orig_mod.' + k: v for k, v in checkpoint_state_dict.items()}

	# Try to load model weights
	try:
		model_to_load.load_state_dict(checkpoint_state_dict)
		print("✅ Model weights loaded successfully.")
	except Exception as e:
		print(f"⚠️ Error loading model state_dict: {e}")
		return 0, None

	# Try to load optimizer
	if 'optimizer_state_dict' in checkpoint and optimizer is not None:
		try:
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			print("✅ Optimizer state loaded successfully.")
		except Exception as e:
			print(f"⚠️ Could not load optimizer state: {e}")

	start_epoch = checkpoint.get('epoch', 0) + 1
	last_loss = checkpoint.get('loss', None)
	print(f"Resumed from epoch {checkpoint.get('epoch', 0)}, last loss: {last_loss}")
	return start_epoch, last_loss