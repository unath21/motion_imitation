# -*- coding: utf-8 -*-
import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import Process, set_start_method
from typing import Dict, Any

from sklearn.cluster import KMeans
from scipy.stats import mode
from typing import Dict, Any, List, Tuple

import csv

from models import create_model, Encoder, Decoder

class SimplePairDataset(Dataset):
    """Returns pairs of images and their masked versions + metadata."""

    IMG_EXTS = {"jpg", "png", "jpeg"}

    def __init__(
        self,
        root: str,
        pairs_path: str,
        frame_size: int = 224,
        split: str = "train",
        seed: int = 42,
        normalization: str = "imagenet",
    ):
        super().__init__()
        assert os.path.isdir(root), f"Root not found: {root}"
        self.root = root
        self.frames_dir = os.path.join(root, "frames")
        self.masked_frames_dir = os.path.join(root, "masked_frames")
        self.frame_size = frame_size
        self.split = split
        self.rng = random.Random(seed)
        self.normalization = normalization
        self.action_idx = {"pushup": 0, "bowl": 1, "bench_press": 2, "pullup": 3, "squat": 4}

        if not os.path.isfile(pairs_path):
            raise FileNotFoundError(f"pairs_path not found: {pairs_path}")
        with open(pairs_path, "r") as f:
            self.pairs = json.load(f)

        required_keys = {"video_id", "i1", "i2", "action"}
        for p in self.pairs:
            if not required_keys.issubset(p.keys()):
                raise ValueError(f"Pair missing required keys: {p}")

        self.video_frames: Dict[str, Any] = {}
        self._index_videos()
    
    def __len__(self):
        return len(self.pairs)

    def _index_videos(self):
        if not os.path.exists(self.frames_dir):
            raise FileNotFoundError(f"Frames directory not found: {self.frames_dir}")

        for vid in sorted(os.listdir(self.frames_dir)):
            vpath = os.path.join(self.frames_dir, vid)
            if not os.path.isdir(vpath):
                continue
            frame_files = sorted(
                [f for f in os.listdir(vpath) if f.split(".")[-1].lower() in self.IMG_EXTS],
                key=lambda x: int(os.path.splitext(x)[0])
            )
            if frame_files:
                self.video_frames[vid] = frame_files

        missing = [p["video_id"] for p in self.pairs if p["video_id"] not in self.video_frames]
        if missing:
            raise RuntimeError(f"Pairs reference missing videos: {sorted(set(missing))[:5]} ...")

    @staticmethod
    def _bgr_to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_frame(self, video_id: str, frame_file: str):
        path = os.path.join(self.frames_dir, video_id, frame_file)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read frame: {path}")
        return self._bgr_to_rgb(img)

    def _load_masked_frame(self, video_id: str, frame_file: str):
        name, ext = os.path.splitext(frame_file)
        frame_num = int(name) - 1
        frame_file_prev = f"{frame_num:06d}{ext}"  # zero-padded to 6 digits
        path = os.path.join(self.masked_frames_dir, video_id, frame_file_prev)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read frame: {path}")
        return self._bgr_to_rgb(img)

    def _apply_augmentations(self, *images):
        if self.split != "train" or not images:
            return images
        augmented = list(images)
        if self.rng.random() < 0.5:
            augmented = [cv2.flip(img, 1) for img in augmented]
        if self.rng.random() < 0.3:
            alpha = 1.0 + self.rng.uniform(-0.1, 0.1)
            beta = self.rng.uniform(-0.05, 0.05) * 255

            def apply_jitter(img):
                out = img.astype("float32") * alpha + beta
                return np.clip(out, 0, 255).astype("uint8")

            augmented = [apply_jitter(img) for img in augmented]
        return tuple(augmented)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        # [H, W, C] uint8 -> [C, H, W] float in [0,1]
        img = img.astype("float32") / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)

        if self.normalization == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype)
        else:  # vae-style: [-1,1]
            mean = torch.tensor([0.5, 0.5, 0.5], dtype=tensor.dtype)
            std = torch.tensor([0.5, 0.5, 0.5], dtype=tensor.dtype)

        tensor = (tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
        return tensor

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        vid = pair["video_id"]
        i1 = pair["i1"]
        i2 = pair["i2"]
        action_idx = self.action_idx[pair["action"]]

        frames = self.video_frames[vid]
        if i1 >= len(frames) or i2 >= len(frames):
            raise IndexError(f"Frame index out of range for video {vid}: {i1},{i2} >= {len(frames)}")

        f1, f2 = frames[i1], frames[i2]
        img1 = self._load_frame(vid, f1)
        img2 = self._load_frame(vid, f2)
        masked_img1 = self._load_masked_frame(vid, f1)
        masked_img2 = self._load_masked_frame(vid, f2)

        # Resize: keep short side = frame_size + 32
        def resize_keep_short(img, target_short):
            h, w = img.shape[:2]
            scale = target_short / min(h, w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        target_short = self.frame_size + 32
        img1 = resize_keep_short(img1, target_short)
        img2 = resize_keep_short(img2, target_short)
        masked_img1 = resize_keep_short(masked_img1, target_short)
        masked_img2 = resize_keep_short(masked_img2, target_short)

        # Crop same region
        h, w = img1.shape[:2]
        if self.split == "train":
            y = self.rng.randint(0, max(0, h - self.frame_size))
            x = self.rng.randint(0, max(0, w - self.frame_size))
        else:
            y = max(0, (h - self.frame_size) // 2)
            x = max(0, (w - self.frame_size) // 2)

        img1 = img1[y:y + self.frame_size, x:x + self.frame_size]
        masked_img1 = masked_img1[y:y + self.frame_size, x:x + self.frame_size]

        h2, w2 = img2.shape[:2]
        y2 = min(y, max(0, h2 - self.frame_size))
        x2 = min(x, max(0, w2 - self.frame_size))
        img2 = img2[y2:y2 + self.frame_size, x2:x2 + self.frame_size]
        masked_img2 = masked_img2[y2:y2 + self.frame_size, x2:x2 + self.frame_size]

        img1, img2, masked_img1, masked_img2 = self._apply_augmentations(img1, img2, masked_img1, masked_img2)

        out: Dict[str, Any] = {
            "img1": self._to_tensor(img1),
            "img2": self._to_tensor(img2),
            "masked_img1": self._to_tensor(masked_img1),
            "masked_img2": self._to_tensor(masked_img2),
            "video_id": torch.tensor(int(vid), dtype=torch.int64),
            "action": torch.tensor(action_idx, dtype=torch.int64),
            "frame_idx1": int(i1),
            "frame_idx2": int(i2),
            "delta": int(i2 - i1),
        }
        return out


def initialize_model(model_type, latent_dim, device):
	if "latent_inputs" in model_type:
		vae_encoder = Encoder.from_pretrained(
			f"stabilityai/stable-diffusion-2-1",
			subfolder="vae",
			torch_dtype=torch.bfloat16,
		).to(device).eval()

		model = create_model(
			name=model_type,
			in_channels=vae_encoder.config.latent_channels,
			out_channels=vae_encoder.config.latent_channels,
			z_channels=latent_dim,
			encoder_block_out_channels=(64, 128, 256),
			decoder_cond_flatten=True if model_type == "latent_inputs_v2" else False,
			decoder_cond_scale=64 if model_type == "latent_inputs_v2" else 1,
		)
		return model.to(device), vae_encoder

	model = create_model(
			name=model_type,
			in_channels=3,
			out_channels=3,
			z_channels=latent_dim,
			encoder_block_out_channels=(64, 128, 256),
			decoder_cond_flatten=True if 'v2' in model_type else False,
			decoder_cond_scale=64 * 64 if 'v2' in model_type else 1,
		)
	return model.to(device), None


# ========= Evaluation helpers =========
@torch.inference_mode()
def get_results(model, vae_encoder, dataloader, device, if_sam, if_latent):
    pooled_latents = []
    pooled_actions = []
    pooled_video_ids = []

    for batch in dataloader:
        i1 = batch["img1"].to(device, non_blocking=True).to(torch.bfloat16)
        i2 = batch["img2"].to(device, non_blocking=True).to(torch.bfloat16)
        action = batch["action"].to(device, non_blocking=True)
        video = batch["video_id"].to(device, non_blocking=True)
        masked_x1 = batch["masked_img1"].to(device, non_blocking=True).to(torch.bfloat16)

        if if_sam:
            _, pooled_z = model(i1, i2, masked_x1)  # assumes (recon, pooled_z)
        else:
            if not if_latent:
                _, pooled_z = model(i1, i2)             # assumes (recon, pooled_z)
            else:
                z1 = vae_encoder(i1).latent_dist.sample()
                z2 = vae_encoder(i2).latent_dist.sample()
                _, pooled_z = model(z1, z2)  # assumes (recon, pooled_z)

        pooled_latents.append(pooled_z)
        pooled_actions.append(action)
        pooled_video_ids.append(video)

    pooled_latents = torch.cat(pooled_latents, dim=0)
    pooled_actions = torch.cat(pooled_actions, dim=0)
    pooled_videos = torch.cat(pooled_video_ids, dim=0)
    return pooled_latents, pooled_actions, pooled_videos


@torch.inference_mode()
def evaluate_kmeans_clustering(latents, actions, action_dict, epoch, output_dir):
    """
    Runs K-Means clustering on latent representations and saves per-class + overall
    accuracies to a single consolidated CSV in output_dir. Also saves raw tensors per epoch.
    """
    num_classes = len(action_dict)
    action_names = list(action_dict.keys())
    os.makedirs(output_dir, exist_ok=True)

    # --- Save raw tensors for this epoch ---
    torch.save(latents.cpu(), os.path.join(output_dir, f"latents_epoch_{epoch}.pt"))
    torch.save(actions.cpu(), os.path.join(output_dir, f"actions_epoch_{epoch}.pt"))

    # --- Prepare & run KMeans ---
    latents_np = latents.float().cpu().numpy()
    actions_np = actions.cpu().numpy()

    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(latents_np)

    # Map clusters to majority true labels
    mapped_labels = np.zeros_like(kmeans_labels)
    for cluster_idx in range(num_classes):
        mask = (kmeans_labels == cluster_idx)
        if np.any(mask):
            majority_label = mode(actions_np[mask], keepdims=True)[0][0]
            mapped_labels[mask] = majority_label

    # Accuracies
    overall_acc = np.mean(mapped_labels == actions_np)
    per_class_acc = {}
    for class_idx, name in enumerate(action_names):
        mask = (actions_np == class_idx)
        per_class_acc[name] = np.mean(mapped_labels[mask] == class_idx) if np.any(mask) else np.nan

    # Append to consolidated CSV
    csv_path = os.path.join(output_dir, "kmeans_epoch_accuracies.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch"] + action_names + ["overall"])
        writer.writerow([epoch] + [per_class_acc[n] for n in action_names] + [overall_acc])

    print(f"[KMeans] Epoch {epoch} | overall={overall_acc:.4f} | per-class={per_class_acc}")
    return overall_acc, per_class_acc


def evaluate_epoch_accuracies(model, vae_encoder, checkpoint_path, epoch, dataloader, action_dict, device, if_sam, if_latent,results_dir):
    print(f"--- Processing Epoch {epoch} ---")
    checkpoint = os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch}.pt")
    if not os.path.exists(checkpoint):
        print(f"[WARN] Missing checkpoint: {checkpoint} (skipping epoch)")
        return None, None

    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    latents, actions, video_ids = get_results(model, vae_encoder, dataloader, device, if_sam, if_latent)

    # --- K-Means clustering (saves CSV + latents/actions) ---
    evaluate_kmeans_clustering(
        latents=latents,
        actions=actions,
        action_dict=action_dict,
        epoch=epoch,
        output_dir=results_dir,
    )

    # Distances on GPU then move to CPU for pandas (supervised nearest-class mean)
    dist_matrix = torch.cdist(latents, latents, p=2.0)
    dist_matrix.fill_diagonal_(float("nan"))

    num_classes = len(action_dict)
    class_mean_cols = []
    for c in range(num_classes):
        class_mask = (actions == c)
        distances_to_c = dist_matrix[:, class_mask]
        mean_dist_c = torch.nanmean(distances_to_c, dim=1)
        class_mean_cols.append(mean_dist_c)

    final_table_tensor = torch.stack(class_mean_cols, dim=1)  # [N, C]

    col_names = list(action_dict.keys())
    df = pd.DataFrame(
        final_table_tensor.float().detach().cpu().numpy(),
        index=video_ids.detach().cpu().tolist(),
        columns=col_names
    )

    reverse_action = {v: k for k, v in action_dict.items()}
    df["original_class"] = [reverse_action[int(a.item())] for a in actions.detach().cpu()]
    dist_cols = [c for c in df.columns if c != "original_class"]
    df["predicted_class"] = df[dist_cols].idxmin(axis=1)
    df["is_correct"] = (df["predicted_class"] == df["original_class"]).astype(int)
    class_accuracies = df.groupby("original_class")["is_correct"].mean().to_dict()
    print(f"Epoch {epoch} Accuracies (nearest-class mean): {class_accuracies}")

    # Return supervised class accuracies and also stash raw tensors for upstream aggregation if desired
    save_dict = {
        "latents": latents.cpu(),
        "actions": actions.cpu(),
        "video_ids": video_ids.cpu(),
    }
    return class_accuracies, save_dict


def run_all_evaluations_and_plot(model, vae_encoder, checkpoint_path, dataloader, action_dict, if_sam, if_latent, plot_filename, device):
    """
    Runs evaluation every 50 epochs from 0..999.
    - Saves KMeans CSV + tensors in results_dir
    - Plots supervised class accuracy (existing behavior)
    - Saves per-epoch latents/actions/video_ids aggregated to a single .pt for convenience
    """
    # Make a results directory tied to (checkpoint_path, dataset/model stem)
    stem = os.path.splitext(os.path.basename(plot_filename))[0]
    results_dir = os.path.join(checkpoint_path, f"eval_{stem}")
    os.makedirs(results_dir, exist_ok=True)

    all_epoch_results: List[Dict[str, float]] = []
    latent_dicts: Dict[int, Dict[str, torch.Tensor]] = {}

    for epoch in range(0, 1000, 50):
        acc, save_dict = evaluate_epoch_accuracies(
            model, vae_encoder, checkpoint_path, epoch, dataloader, action_dict, device, if_sam, if_latent, results_dir
        )
        if acc is None:
            continue
        latent_dicts[epoch] = save_dict
        acc["epoch"] = epoch
        all_epoch_results.append(acc)

    if not all_epoch_results:
        print("No results to plot.")
        return

    results_df = pd.DataFrame(all_epoch_results).set_index("epoch").sort_index()
    print("\n--- Final Accuracy Results Table (nearest-class mean) ---")
    print(results_df.to_string())

    # Save latent_dicts for later use (supervised eval tensors)
    torch.save(latent_dicts, os.path.join(results_dir, f"{stem}_latents_all_epochs.pt"))

    # Save supervised accuracy CSV + plot
    sup_csv = os.path.join(results_dir, f"{stem}_supervised_accuracies.csv")
    results_df.to_csv(sup_csv)

    plt.figure(figsize=(12, 7))
    ax = results_df.plot(kind="line", marker="o", figsize=(12, 7))
    plt.title("Class Accuracy vs. Epoch (Nearest-Class Mean)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--")
    plt.tight_layout()
    plot_path = os.path.join(results_dir, plot_filename)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Supervised accuracy plot saved to '{plot_path}'")
    print(f"KMeans CSV lives at: {os.path.join(results_dir, 'kmeans_epoch_accuracies.csv')}")


# ========= Multi-GPU worker =========
def worker(gpu_id: int, model_name: str, latent_dim: str, checkpoint_dir: str, data_root: str, dataset_paths, batch_size: int):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu_id)

    # Build datasets & loaders (val split)
    val_datasets = [
        SimplePairDataset(root=data_root, pairs_path=pp, frame_size=256, split="val", normalization="default" if "latent" in model_name else "imagenet")
        for pp in dataset_paths
    ]
    val_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        for ds in val_datasets
    ]

    model, vae_encoder = initialize_model(model_name, latent_dim, device)

    action_dict = {"pushup": 0, "bowl": 1, "bench_press": 2, "pullup": 3, "squat": 4}

    for val_loader, dataset_path in zip(val_loaders, dataset_paths):
        base = os.path.basename(dataset_path)
        stem = os.path.splitext(base)[0]
        plot_filename = f"{model_name}_accuracy_plot_{stem}.png"

        run_all_evaluations_and_plot(
            model=model.to(torch.bfloat16),
            vae_encoder=vae_encoder,
            checkpoint_path=checkpoint_dir,
            dataloader=val_loader,
            action_dict=action_dict,
            if_sam="masked_inputs" in model_name,
            if_latent="latent_inputs" in model_name,
            plot_filename=plot_filename,
            device=device,
        )


# ========= Main (spawn one process per model) =========
def main():
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/rgoel15/data_pturaga/datasets/Penn_Action")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    dataset_paths = [
        "/home/rgoel15/motion_imitation/data/jsons/evaluation_set_train.json",
        # "/home/rgoel15/motion_imitation/data/jsons/train_set.json",
        # "/home/rgoel15/motion_imitation/data/jsons/val_set.json",
    ]

    models = {
        # "masked_inputs_v2": {"latent_dim": 2, "checkpoint": "/scratch/rgoel15/motion_emitation_data/model_image_diff_z_masked_inputs_without_pooling_channel_2"},
        # "masked_inputs": {"latent_dim": 64, "checkpoint": "/scratch/rgoel15/motion_emitation_data/model_image_diff_z_with_masked_inputs"},
        # "latent_inputs_v2": {"latent_dim": 4, "checkpoint": "/scratch/rgoel15/motion_emitation_data/model_latent_input_diff_z_without_pooling_c_4"},
        # "latent_inputs": {"latent_dim": 4, "checkpoint": "/scratch/rgoel15/motion_emitation_data/model_latent_input_diff_z_with_pooling_c_4"},
        # "autoencoder": {"latent_dim": 64, "checkpoint": "/scratch/rgoel15/motion_emitation_data/model_A_cleaned_const_delta_4"},
        "autoencoder_v2": {"latent_dim": 1, "checkpoint": "/home/rgoel15/scratch/motion_emitation_data/model_image_diff_z_without_pooling_channel_1"},
        # "autoencoder_v2": {"latent_dim": 2, "checkpoint": "/home/rgoel15/scratch/motion_emitation_data/model_image_diff_z_without_pooling_channel_2"},
    }

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices available.")

    print(f"Found {num_gpus} GPUs and {len(models)} models to evaluate.")
    
    # Convert models dict to list of tuples for easier processing
    model_list = list(models.items())
    completed = 0
    total_models = len(model_list)
    
    # Process models in batches
    while completed < total_models:
        # Determine how many models to process in this batch
        batch_size = min(num_gpus, total_models - completed)
        current_batch = model_list[completed:completed + batch_size]
        
        print(f"\n=== Processing batch {completed//num_gpus + 1}: Models {completed+1}-{completed+batch_size} of {total_models} ===")
        
        procs = []
        for idx, (model_name, model_info) in enumerate(current_batch):
            gpu_id = idx  # Use GPU 0, 1, 2, ... for this batch
            print(f"Starting {model_name} on GPU {gpu_id}")
            
            p = Process(
                target=worker, 
                args=(
                    gpu_id, 
                    model_name, 
                    model_info["latent_dim"], 
                    model_info["checkpoint"], 
                    args.data_root, 
                    dataset_paths, 
                    args.batch_size
                )
            )
            p.start()
            procs.append(p)
        
        # Wait for all processes in this batch to complete
        for p in procs:
            p.join()
        
        completed += batch_size
        print(f"=== Batch complete. {completed}/{total_models} models finished ===")

    print("[DONE] All model evaluations completed.")


if __name__ == "__main__":
    main()