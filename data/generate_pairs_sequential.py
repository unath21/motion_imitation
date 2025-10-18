#!/usr/bin/env python3
"""Generate fixed (video_id, i1, i2, delta) pairs for Penn Action.

Outputs JSON lists for train and validation splits for reproducible experiments.

Usage (example):
  python generate_pairs.py \
    --root /scratch/unath/motion_emitation_data/Penn_Action \
    --train-pairs-out data/train_pairs.json \
    --val-pairs-out data/val_pairs.json \
    --train-pairs 50000 --val-pairs 10000 \
    --delta-min 1 --delta-max 4 --seed 42

JSON format:
[
  {"video_id": "0001", "i1": 12, "i2": 15, "delta": 3},
  ...
]

Notes:
- train_flag in .mat is assumed: 1=train, -1=test (validation).
- Sampling is random but seeded for reproducibility.
- Videos too short for requested delta range are skipped automatically.
- Duplicates are allowed (not de-duplicated) as random resampling; typical for large pair counts.

Potential Extensions:
- Add uniqueness constraints per video.
- Balance deltas exactly across range.
- Curriculum generation (smaller deltas first, then larger).
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from typing import List, Dict, Any

import scipy.io as sio

# -----------------------------------------------------

def scan_videos(root: str):
    frames_dir = os.path.join(root, 'frames')
    labels_dir = os.path.join(root, 'labels')
    videos = []
    for vid in sorted(os.listdir(frames_dir)):
        vpath = os.path.join(frames_dir, vid)
        if not os.path.isdir(vpath):
            continue
        label_path = os.path.join(labels_dir, f'{vid}.mat')
        if not os.path.isfile(label_path):
            continue
        try:
            mat = sio.loadmat(label_path)
            train_flag = int(mat.get('train', [[1]])[0][0])  # 1 train, -1 val/test
            action = str(mat.get('action', [''])[0])
        except Exception:
            continue
        frame_files = [f for f in os.listdir(vpath) if f.lower().endswith('.jpg')]
        frame_files.sort()
        n = len(frame_files)
        if n < 2:
            continue
        videos.append({
            'video_id': vid,
            'num_frames': n,
            'train_flag': train_flag,
            'action': action,
        })
    return videos

# -----------------------------------------------------

def sample_pairs(videos: List[Dict[str, Any]], num_pairs: int, delta_min: int, delta_max: int, rng: random.Random, balance_delta: bool=False) -> List[Dict[str, int]]:
    # Separate videos by length sufficiency for given deltas.
    usable = []
    for v in videos:
        if v['num_frames'] >= delta_min + 1:
            usable.append(v)
    if not usable:
        raise RuntimeError('No usable videos found for given delta range.')

    pairs: List[Dict[str, int]] = []

    for i, vid in enumerate(usable):
        n = vid['num_frames']

        for j in range(n - delta_min):
            i1 = j
            i2 = j + delta_min
            pairs.append({
                'video_id': vid['video_id'],
                'i1': i1,
                'i2': i2,
                'delta': delta_min,
                'action': vid['action'],
            })

    return pairs

# -----------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Generate PennAction train/val frame pairs.')
    ap.add_argument('--root', required=True, help='PennAction root directory (contains frames/, labels/)')
    ap.add_argument('--train-pairs-out', required=True, help='Output JSON for train pairs')
    ap.add_argument('--val-pairs-out', required=True, help='Output JSON for val pairs')
    ap.add_argument('--train-pairs', type=int, default=800000)
    ap.add_argument('--val-pairs', type=int, default=100000)
    ap.add_argument('--delta-min', type=int, default=4)
    ap.add_argument('--delta-max', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--action-list', nargs='+', default=['squat', 'pushup', 'bowl', 'pullup', 'baseball_pitch', 'bench_press', 'situp', 'jumping_jacks'], help='List of actions to include')
    ap.add_argument('--balance-delta', action='store_true', help='Cycle deltas to approximate uniform distribution')
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print('[INFO] Scanning videos...')
    action_list = args.action_list
    videos = scan_videos(args.root)
    train_videos = [v for v in videos if v['train_flag'] == 1]
    val_videos = [v for v in videos if v['train_flag'] != 1]

    print(f'[INFO] Found {len(train_videos)} train videos, {len(val_videos)} val videos.')

    print('[INFO] Sampling train pairs...')
    train_pairs = sample_pairs(train_videos, args.train_pairs, args.delta_min, args.delta_max, rng, args.balance_delta)
    print('[INFO] Sampling val pairs...')
    val_pairs = sample_pairs(val_videos, args.val_pairs, args.delta_min, args.delta_max, rng, args.balance_delta)

    os.makedirs(os.path.dirname(args.train_pairs_out) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.val_pairs_out) or '.', exist_ok=True)

    with open(args.train_pairs_out, 'w') as f:
        json.dump(train_pairs, f)
    with open(args.val_pairs_out, 'w') as f:
        json.dump(val_pairs, f)

    # Basic stats
    def stats(pairs, name):
        if not pairs:
            return
        deltas = [p['delta'] for p in pairs]
        avg_delta = sum(deltas) / len(deltas)
        print(f'[STATS] {name}: count={len(pairs)} avg_delta={avg_delta:.2f} min_delta={min(deltas)} max_delta={max(deltas)}')

    stats(train_pairs, 'train')
    stats(val_pairs, 'val')

    print('[DONE] Saved:')
    print('  Train ->', args.train_pairs_out)
    print('  Val   ->', args.val_pairs_out)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted.', file=sys.stderr)
        sys.exit(1)
