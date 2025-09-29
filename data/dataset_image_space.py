import os
import random
from typing import List, Dict, Any, Optional
import json

import cv2
import torch
from torch.utils.data import Dataset

# Basic joint transforms: we implement minimal deterministic geometry + stochastic color jitter option.
# Additional augmentations (affine, more color ops) can be added later.


class PennActionImagePairDataset(Dataset):
    """Fixed-pair PennAction dataset (no dynamic temporal sampling).

    Expects a JSON list of dicts with keys:
        {"video_id": str, "i1": int, "i2": int, "delta": int}

    Only loads the frames required for each pair.
    Output tensors are normalized to [-1,1].
    Augmentations (random crop region selection, hflip, color jitter) only applied when split == 'train'.
    """

    IMG_EXTS = {"jpg", "png", "jpeg"}

    def __init__(
        self,
        root: str,
        pairs_path: str,
        frame_size: int = 224,
        split: str = 'train',
        color_jitter_prob: float = 0.0,
        horizontal_flip_prob: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        assert os.path.isdir(root), f"Root not found: {root}"
        self.root = root
        self.frames_dir = os.path.join(root, "frames")
        self.frame_size = frame_size
        self.split = split
        self.color_jitter_prob = color_jitter_prob
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rng = random.Random(seed)
        if not os.path.isfile(pairs_path):
            raise FileNotFoundError(f"pairs_path not found: {pairs_path}")
        with open(pairs_path, 'r') as f:
            self.pairs = json.load(f)
        required_keys = {"video_id", "i1", "i2", "delta"}
        for p in self.pairs:
            if not required_keys.issubset(p.keys()):
                raise ValueError(f"Pair missing required keys: {p}")

        # Build a simple mapping of video_id -> list of frame filenames
        self.video_frames = {}
        self._index_videos()
        # Pre-create a cache for video frame lists (already loaded) and optionally frame images later
        self._vid_exists = set(self.video_frames.keys())

    # --------------------------------------------------
    def _index_videos(self):
        videos = sorted(os.listdir(self.frames_dir))
        for vid in videos:
            vpath = os.path.join(self.frames_dir, vid)
            if not os.path.isdir(vpath):
                continue
            frame_files = sorted([
                f for f in os.listdir(vpath)
                if f.split('.')[-1].lower() in self.IMG_EXTS
            ])
            if frame_files:
                self.video_frames[vid] = frame_files
        # Validate that all pair video_ids exist
        missing = [p['video_id'] for p in self.pairs if p['video_id'] not in self.video_frames]
        if missing:
            raise RuntimeError(f"Pairs reference missing videos: {sorted(set(missing))[:5]} ...")

    # --------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    # --------------------------------------------------
    # (Dynamic sampling helpers removed – fixed pairs only)

    # --------------------------------------------------
    def _load_frame(self, video_id: str, frame_file: str):
        path = os.path.join(self.frames_dir, video_id, frame_file)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read frame: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # --------------------------------------------------
    def _resize_and_crop_pair(self, img1, img2):
        # Resize so shorter side = 256 then random or center crop 224
        def resize(img):
            h, w = img.shape[:2]
            scale = 256.0 / min(h, w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        r1 = resize(img1)
        r2 = resize(img2)
        h, w = r1.shape[:2]
        ch = cw = self.frame_size
        if self.split == 'train':
            if h == ch and w == cw:
                y = x = 0
            else:
                y = self.rng.randint(0, max(0, h - ch))
                x = self.rng.randint(0, max(0, w - cw))
        else:
            y = max(0, (h - ch) // 2)
            x = max(0, (w - cw) // 2)
        c1 = r1[y:y+ch, x:x+cw]
        c2 = r2[y:y+ch, x:x+cw]
        return c1, c2

    # --------------------------------------------------
    def _maybe_hflip(self, img1, img2):
        if self.split == 'train' and self.rng.random() < self.horizontal_flip_prob:
            return cv2.flip(img1, 1), cv2.flip(img2, 1)
        return img1, img2

    # --------------------------------------------------
    def _maybe_color_jitter_pair(self, img1, img2):
        """Apply the SAME simple brightness/contrast jitter to both frames.

        Rationale: keep photometric consistency across the pair while still
        preventing the network from overfitting to absolute lighting, but NOT
        introducing artificial discrepancies that could mimic motion.
        """
        if (
            self.split != 'train'
            or self.color_jitter_prob <= 0
            or self.rng.random() > self.color_jitter_prob
        ):
            return img1, img2
        alpha = 1.0 + self.rng.uniform(-0.2, 0.2)  # contrast factor
        beta = self.rng.uniform(-0.1, 0.1) * 255    # brightness shift
        def apply(img):
            out = img.astype('float32') * alpha + beta
            return out.clip(0, 255).astype('uint8')
        return apply(img1), apply(img2)

    # --------------------------------------------------
    def _to_tensor(self, img):
        # img uint8 RGB HxWx3 -> tensor float [-1,1]
        img = img.astype('float32') / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W
        tensor = tensor * 2.0 - 1.0
        return tensor

    # --------------------------------------------------
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        vid = pair['video_id']
        i1 = pair['i1']
        i2 = pair['i2']
        delta = pair['delta']
        frames = self.video_frames[vid]
        # Basic bounds check (robust against outdated pairs)
        if i1 >= len(frames) or i2 >= len(frames):
            raise IndexError(f"Frame index out of range for video {vid}: {i1},{i2} >= {len(frames)}")
        f1 = frames[i1]
        f2 = frames[i2]
        img1 = self._load_frame(vid, f1)
        img2 = self._load_frame(vid, f2)

        img1, img2 = self._resize_and_crop_pair(img1, img2)
        img1, img2 = self._maybe_hflip(img1, img2)
        # Apply identical color jitter to both frames (photometric consistency)
        img1, img2 = self._maybe_color_jitter_pair(img1, img2)

        t1 = self._to_tensor(img1)
        t2 = self._to_tensor(img2)

        sample = {
            'I1': t1,
            'I2': t2,
            'delta': int(delta),
            'video_id': vid,
            'frame_index': int(i1),
            'frame_index_2': int(i2),
        }
        return sample


if __name__ == '__main__':
    # Example manual smoke (requires existing pairs JSON)
    ds_root = '/scratch/unath/motion_emitation_data/Penn_Action'
    pairs_json = 'data/train_pairs.json'
    if os.path.isdir(ds_root) and os.path.isfile(pairs_json):
        ds = PennActionImagePairDataset(ds_root, pairs_path=pairs_json)
        item = ds[0]
        print('Sample keys:', item.keys())
        print('I1 shape:', item['I1'].shape, 'I2 shape:', item['I2'].shape, 'delta:', item['delta'])
    else:
        print('Provide valid dataset root and pairs JSON for smoke test.')
