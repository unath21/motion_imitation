import os
import random
from typing import List, Dict, Any, Optional
import json

import cv2
import torch
from torch.utils.data import Dataset


class SimplePairDataset(Dataset):
    """Simple image pair dataset for SimpleViT training.
    
    Returns pairs of images (I1, I2) for motion-aware image generation.
    Compatible with Penn_Action dataset structure but simplified for training.
    """

    IMG_EXTS = {"jpg", "png", "jpeg"}

    def __init__(
        self,
        root: str,
        pairs_path: str,
        frame_size: int = 224,
        split: str = 'train',
        seed: int = 42,
    ):
        """
        Args:
            root: Path to dataset root (contains 'frames' directory)
            pairs_path: Path to JSON file with frame pairs
            frame_size: Size to resize images to (square)
            split: 'train' or 'val' (affects augmentations)
            seed: Random seed for reproducibility
        """
        super().__init__()
        assert os.path.isdir(root), f"Root not found: {root}"
        self.root = root
        self.frames_dir = os.path.join(root, "frames")
        self.frame_size = frame_size
        self.split = split
        self.rng = random.Random(seed)
        
        # Load pairs
        if not os.path.isfile(pairs_path):
            raise FileNotFoundError(f"pairs_path not found: {pairs_path}")
        with open(pairs_path, 'r') as f:
            self.pairs = json.load(f)
        
        # Validate pair format
        required_keys = {"video_id", "i1", "i2"}
        for p in self.pairs:
            if not required_keys.issubset(p.keys()):
                raise ValueError(f"Pair missing required keys: {p}")

        # Index video frames
        self.video_frames = {}
        self._index_videos()

    def _index_videos(self):
        """Build mapping of video_id -> sorted list of frame filenames"""
        if not os.path.exists(self.frames_dir):
            raise FileNotFoundError(f"Frames directory not found: {self.frames_dir}")
            
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

    def __len__(self):
        return len(self.pairs)

    def _load_frame(self, video_id: str, frame_file: str):
        """Load a single frame from disk"""
        path = os.path.join(self.frames_dir, video_id, frame_file)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read frame: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _resize_and_crop(self, img):
        """Resize and crop image to target size"""
        h, w = img.shape[:2]
        
        # Resize to make shorter side = frame_size + 32 for crop margin
        target_size = self.frame_size + 32
        scale = target_size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # Crop to exact frame_size
        h, w = img.shape[:2]
        if self.split == 'train':
            # Random crop for training
            y = self.rng.randint(0, max(0, h - self.frame_size))
            x = self.rng.randint(0, max(0, w - self.frame_size))
        else:
            # Center crop for validation
            y = max(0, (h - self.frame_size) // 2)
            x = max(0, (w - self.frame_size) // 2)
        
        cropped = img[y:y+self.frame_size, x:x+self.frame_size]
        return cropped

    def _apply_augmentations(self, img1, img2):
        """Apply simple augmentations for training"""
        if self.split != 'train':
            return img1, img2
        
        # Horizontal flip (same for both images)
        if self.rng.random() < 0.5:
            img1 = cv2.flip(img1, 1)
            img2 = cv2.flip(img2, 1)
        
        # Simple color jitter (same for both to maintain consistency)
        if self.rng.random() < 0.3:
            alpha = 1.0 + self.rng.uniform(-0.1, 0.1)  # contrast
            beta = self.rng.uniform(-0.05, 0.05) * 255  # brightness
            
            def apply_jitter(img):
                out = img.astype('float32') * alpha + beta
                return out.clip(0, 255).astype('uint8')
            
            img1 = apply_jitter(img1)
            img2 = apply_jitter(img2)
        
        return img1, img2

    def _to_tensor(self, img):
        """Convert numpy image to tensor with ImageNet normalization"""
        # Convert to float and normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        # Apply ImageNet normalization
        # ImageNet mean: [0.485, 0.456, 0.406] for RGB
        # ImageNet std: [0.229, 0.224, 0.225] for RGB
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype, device=tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype, device=tensor.device)
        
        # Normalize: (x - mean) / std
        tensor = (tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
        
        return tensor

    def __getitem__(self, idx):
        """Get a pair of images"""
        pair = self.pairs[idx]
        vid = pair['video_id']
        i1 = pair['i1']
        i2 = pair['i2']
        
        frames = self.video_frames[vid]
        
        # Bounds check
        if i1 >= len(frames) or i2 >= len(frames):
            raise IndexError(f"Frame index out of range for video {vid}: {i1},{i2} >= {len(frames)}")
        
        # Load frames
        f1 = frames[i1]
        f2 = frames[i2]
        img1 = self._load_frame(vid, f1)
        img2 = self._load_frame(vid, f2)
        
        # Process images with same crop region to maintain spatial consistency
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Use the same resize/crop parameters for both images
        target_size = self.frame_size + 32
        scale1 = target_size / min(h1, w1)
        scale2 = target_size / min(h2, w2)
        
        # Resize both
        nh1, nw1 = int(round(h1 * scale1)), int(round(w1 * scale1))
        nh2, nw2 = int(round(h2 * scale2)), int(round(w2 * scale2))
        img1 = cv2.resize(img1, (nw1, nh1), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (nw2, nh2), interpolation=cv2.INTER_LINEAR)
        
        # Apply same crop coordinates (use img1's dimensions as reference)
        h, w = img1.shape[:2]
        if self.split == 'train':
            y = self.rng.randint(0, max(0, h - self.frame_size))
            x = self.rng.randint(0, max(0, w - self.frame_size))
        else:
            y = max(0, (h - self.frame_size) // 2)
            x = max(0, (w - self.frame_size) // 2)
        
        # Crop both images
        img1 = img1[y:y+self.frame_size, x:x+self.frame_size]
        
        # For img2, adjust crop coordinates if needed
        h2, w2 = img2.shape[:2]
        y2 = min(y, max(0, h2 - self.frame_size))
        x2 = min(x, max(0, w2 - self.frame_size))
        img2 = img2[y2:y2+self.frame_size, x2:x2+self.frame_size]
        
        # Apply augmentations
        img1, img2 = self._apply_augmentations(img1, img2)
        
        # Convert to tensors
        tensor1 = self._to_tensor(img1)
        tensor2 = self._to_tensor(img2)
        
        return {
            'img1': tensor1,           # First image
            'img2': tensor2,           # Second image  
            'video_id': vid,           # Video identifier
            'frame_idx1': int(i1),     # First frame index
            'frame_idx2': int(i2),     # Second frame index
            'delta': int(i2 - i1),     # Frame difference
        }


def create_dummy_pairs(frames_dir: str, output_path: str, max_delta: int = 10, pairs_per_video: int = 5):
    """Create a dummy pairs JSON file for testing"""
    pairs = []
    
    if not os.path.exists(frames_dir):
        print(f"Frames directory not found: {frames_dir}")
        return
    
    videos = sorted([v for v in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, v))])
    
    for vid in videos[:10]:  # Limit to first 10 videos for testing
        vpath = os.path.join(frames_dir, vid)
        frame_files = sorted([
            f for f in os.listdir(vpath)
            if f.split('.')[-1].lower() in {'jpg', 'png', 'jpeg'}
        ])
        
        if len(frame_files) < 2:
            continue
            
        # Create random pairs for this video
        for _ in range(pairs_per_video):
            i1 = random.randint(0, len(frame_files) - max_delta - 1)
            i2 = random.randint(i1 + 1, min(i1 + max_delta, len(frame_files) - 1))
            
            pairs.append({
                'video_id': vid,
                'i1': i1,
                'i2': i2,
                'delta': i2 - i1
            })
    
    with open(output_path, 'w') as f:
        json.dump(pairs, f, indent=2)
    
    print(f"Created {len(pairs)} pairs saved to {output_path}")


if __name__ == '__main__':
    # Test the dataset
    
    # Example paths - adjust as needed
    ds_root = '/scratch/unath/motion_emitation_data/Penn_Action'
    pairs_json = '/home/unath/motion_emitation/MAE/test_pairs.json'
    
    # Create dummy pairs if they don't exist
    if not os.path.exists(pairs_json):
        frames_dir = os.path.join(ds_root, 'frames')
        if os.path.exists(frames_dir):
            create_dummy_pairs(frames_dir, pairs_json)
        else:
            print(f"Cannot create test pairs: {frames_dir} not found")
            exit(1)
    
    # Test dataset
    if os.path.isdir(ds_root) and os.path.isfile(pairs_json):
        print("Testing SimplePairDataset...")
        ds = SimplePairDataset(
            root=ds_root,
            pairs_path=pairs_json,
            frame_size=224,
            split='train'
        )
        
        print(f"Dataset size: {len(ds)}")
        
        # Test loading a sample
        item = ds[0]
        print('Sample keys:', item.keys())
        print('img1 shape:', item['img1'].shape, 'range:', [item['img1'].min().item(), item['img1'].max().item()])
        print('img2 shape:', item['img2'].shape, 'range:', [item['img2'].min().item(), item['img2'].max().item()])
        print('delta:', item['delta'], 'video_id:', item['video_id'])
        
        # Test with your SimpleViT
        print("\nTesting with SimpleViT...")
        import sys
        sys.path.append('/home/unath/motion_emitation/MAE')
        from model_mimic import SimpleViT
        
        model = SimpleViT(image_size=224, patch_size=16, emb_dim=192)
        
        img1 = item['img1'].unsqueeze(0)  # Add batch dimension
        img2 = item['img2'].unsqueeze(0)
        
        print(f"Input shapes - img1: {img1.shape}, img2: {img2.shape}")
        
        with torch.no_grad():
            pred_img2 = model(img1, img2)
            print(f"Output shape: {pred_img2.shape}")
            print("✅ SimpleViT forward pass successful!")
            
    else:
        print('Please provide valid dataset root and create pairs JSON for testing.')
        print(f"Expected: root={ds_root}, pairs={pairs_json}")
