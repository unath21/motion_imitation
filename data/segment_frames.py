# import os
# import cv2
# import numpy as np
# import torch
# import scipy.io as sio
# from PIL import Image
# from transformers import Sam2VideoModel, Sam2VideoProcessor

# def load_frames(data_root, video_id):
#     frames_dir = os.path.join(data_root, 'frames', video_id)
#     files = os.listdir(frames_dir)

#     # ✅ Sort numerically based on frame number
#     files = sorted(files, key=lambda x: int(x.split('.')[0]))
#     frames = []
#     for file in files:
#         frame_path = os.path.join(frames_dir, file)
#         frame = np.array(Image.open(frame_path).convert("RGB"))
#         frames.append(frame)

#     return np.array(frames), frames

# def generate_and_save_masks(video_id, data_root, model, processor, device, save_dir="output_masks"):
#     """
#     Generate segmentation masks for each frame using SAM2 video model,
#     save the masks and masked images (white background only).

#     Args:
#         video_id: ID of the video to process
#         data_root: root directory containing 'frames' and 'labels'
#         model: SAM2VideoModel
#         processor: SAM2VideoProcessor
#         device: torch device
#         save_dir: base directory for saving results
#     """
#     # --- Load bounding box info ---
#     labels_dir = os.path.join(data_root, 'labels')
#     label_path = os.path.join(labels_dir, f"{video_id}.mat")
#     mat = sio.loadmat(label_path)
#     bboxes = mat['bbox']
#     bbox1 = bboxes[0]

#     # --- Load video frames ---
#     frames, pil_frames = load_frames(data_root, video_id)
#     total_frames = len(frames)
#     print(f"[INFO] Total frames in {video_id}: {total_frames}")

#     # --- Create output directories ---
#     mask_dir = os.path.join(save_dir, "masks", video_id)
#     masked_img_dir = os.path.join(save_dir, "masked_frames", video_id)
#     os.makedirs(mask_dir, exist_ok=True)
#     os.makedirs(masked_img_dir, exist_ok=True)

#     # --- Initialize SAM2 video inference session ---
#     inference_session = processor.init_video_session(
#         video=frames,
#         inference_device=device,
#         dtype=torch.bfloat16,
#     )

#     # Add first-frame bbox as input
#     processor.add_inputs_to_inference_session(
#         inference_session=inference_session,
#         frame_idx=0,
#         obj_ids=1,
#         input_boxes=[[[float(bbox1[0]), float(bbox1[1]), float(bbox1[2]), float(bbox1[3])]]]
#     )

#     # Run segmentation on first frame
#     model(inference_session=inference_session, frame_idx=0)

#     # --- Process and save results for all frames ---
#     for sam2_video_output in model.propagate_in_video_iterator(inference_session):
#         frame_idx = sam2_video_output.frame_idx
#         height, width = inference_session.video_height, inference_session.video_width

#         # Get post-processed binary mask
#         masks = processor.post_process_masks(
#             [sam2_video_output.pred_masks],
#             original_sizes=[[height, width]],
#             binarize=True
#         )[0]
#         mask = masks[0][0].numpy().astype(np.uint8) * 255  # Convert to 0/255 uint8

#         # Save mask
#         mask_path = os.path.join(mask_dir, f"{frame_idx:06}.png")
#         cv2.imwrite(mask_path, mask)

#         # Apply mask to frame (make background white)
#         frame = frames[frame_idx]
#         white_bg = np.ones_like(frame, dtype=np.uint8) * 255
#         masked_image = np.where(mask[..., None] > 0, frame, white_bg)

#         # Save masked image
#         masked_img_path = os.path.join(masked_img_dir, f"{frame_idx:06}.jpg")
#         cv2.imwrite(masked_img_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

#     print(f"[DONE] Saved masks and masked frames for {video_id} to {os.path.join(save_dir, video_id)}")

# if __name__ == "__main__":
#     model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-large").to(device, dtype=torch.bfloat16)
#     processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-large")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     data_root = "/home/rgoel15/data_pturaga/datasets/Penn_Action"

#     video_ids = os.listdir(os.path.join(data_root, 'frames'))
#     for video_id in video_ids:
#         generate_and_save_masks(video_id, data_root, model, processor, device)

import os
import cv2
import numpy as np
import torch
import scipy.io as sio
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor
from torch.multiprocessing import Process, set_start_method

def load_frames(data_root, video_id):
    frames_dir = os.path.join(data_root, 'frames', video_id)
    files = sorted(os.listdir(frames_dir), key=lambda x: int(x.split('.')[0]))
    frames = [np.array(Image.open(os.path.join(frames_dir, f)).convert("RGB")) for f in files]
    return np.array(frames), frames


def generate_and_save_masks(video_id, data_root, model, processor, device, save_dir="/data/pturaga/datasets/Penn_Action"):
    """
    Generate segmentation masks for each frame using SAM2 video model,
    save the masks and masked images (white background only).
    """
    labels_dir = os.path.join(data_root, 'labels')
    label_path = os.path.join(labels_dir, f"{video_id}.mat")

    if not os.path.exists(label_path):
        print(f"[WARNING] No label file for {video_id}, skipping.")
        return

    mat = sio.loadmat(label_path)
    bboxes = mat['bbox']
    bbox1 = bboxes[0]

    frames, pil_frames = load_frames(data_root, video_id)
    total_frames = len(frames)
    print(f"[GPU {device.index}] Processing {video_id} ({total_frames} frames)")

    mask_dir = os.path.join(save_dir, "masks", video_id)
    masked_img_dir = os.path.join(save_dir, "masked_frames", video_id)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(masked_img_dir, exist_ok=True)

    inference_session = processor.init_video_session(
        video=frames,
        inference_device=device,
        dtype=torch.bfloat16,
    )

    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=0,
        obj_ids=1,
        input_boxes=[[[float(bbox1[0]), float(bbox1[1]), float(bbox1[2]), float(bbox1[3])]]]
    )

    model(inference_session=inference_session, frame_idx=0)

    for sam2_video_output in model.propagate_in_video_iterator(inference_session):
        frame_idx = sam2_video_output.frame_idx
        height, width = inference_session.video_height, inference_session.video_width

        masks = processor.post_process_masks(
            [sam2_video_output.pred_masks],
            original_sizes=[[height, width]],
            binarize=True
        )[0]
        mask = masks[0][0].cpu().numpy().astype(np.uint8) * 255

        mask_path = os.path.join(mask_dir, f"{frame_idx:06}.png")
        cv2.imwrite(mask_path, mask)

        frame = frames[frame_idx]
        white_bg = np.ones_like(frame, dtype=np.uint8) * 255
        masked_image = np.where(mask[..., None] > 0, frame, white_bg)

        masked_img_path = os.path.join(masked_img_dir, f"{frame_idx:06}.jpg")
        cv2.imwrite(masked_img_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

    print(f"[GPU {device.index}] Done with {video_id}")


def worker(gpu_id, video_ids, data_root, save_dir):
    """
    Worker process that runs on a specific GPU and processes a subset of videos.
    """
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[INFO] Starting worker on GPU {gpu_id} ({len(video_ids)} videos)")

    model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-large").to(device, dtype=torch.bfloat16)
    processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-large")

    for vid in video_ids:
        try:
            generate_and_save_masks(vid, data_root, model, processor, device, save_dir)
        except Exception as e:
            print(f"[ERROR] Failed {vid} on GPU {gpu_id}: {e}")


if __name__ == "__main__":
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    data_root = "/home/rgoel15/data_pturaga/datasets/Penn_Action"
    save_dir = "/data/pturaga/datasets/Penn_Action"

    all_videos = sorted(os.listdir(os.path.join(data_root, "frames")))
    num_gpus = torch.cuda.device_count()
    print(f"[INFO] Found {num_gpus} GPUs, distributing videos...")

    # Split video list evenly across GPUs
    chunks = np.array_split(all_videos, num_gpus)

    # Spawn one process per GPU
    processes = []
    for gpu_id, vids in enumerate(chunks):
        p = Process(target=worker, args=(gpu_id, vids, data_root, save_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("[DONE] All GPUs have finished processing.")
