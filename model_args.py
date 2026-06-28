# Explanation of generator_args is in sam/segment_anything/automatic_mask_generator.py: SamAutomaticMaskGenerator
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "ckpt")

if not os.path.exists(CKPT_DIR):
    raise RuntimeError(f"Checkpoint folder not found: {CKPT_DIR}")

def find_sam_checkpoint(folder=CKPT_DIR):
    files = glob.glob(os.path.join(folder, "*.pth"))

    for f in files:
        name = os.path.basename(f).lower()
        if name.startswith("sam_"):
            return f

    raise FileNotFoundError("No SAM checkpoint found inside ckpt folder")

def infer_sam_type(path):
    name = os.path.basename(path).lower()

    if "vit_h" in name:
        return "vit_h"
    if "vit_l" in name:
        return "vit_l"
    if "vit_b" in name:
        return "vit_b"

    raise ValueError(f"Unknown SAM model type in filename: {name}")

def find_deaot_checkpoint(folder=CKPT_DIR):
    files = glob.glob(os.path.join(folder, "*.pth"))

    for f in files:
        name = os.path.basename(f).lower()

        if "deaot" in name and "pre_ytb_dav" in name:
            return f

    raise FileNotFoundError(
        "No DeAOT PRE_YTB_DAV checkpoint found in ckpt folder"
    )

def infer_deaot_model(path):
    name = os.path.basename(path).lower()

    # size
    if "deaott" in name:
        size = "t"
    elif "deaots" in name:
        size = "s"
    elif "deaotb" in name:
        size = "b"
    else:
        size = "l"

    # backbone
    if "r50" in name:
        backbone = "r50"
    elif "swin" in name:
        backbone = "swinb"
    else:
        backbone = "r50"

    return f"{backbone}_deaot{size}"

sam_ckpt = find_sam_checkpoint()
deaot_ckpt = find_deaot_checkpoint()

sam_args = {
    'sam_checkpoint': sam_ckpt,
    'model_type': infer_sam_type(sam_ckpt),
    'generator_args':{
        'points_per_side': 16,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
    },
    "device": "cuda",
}
aot_args = {
    'phase': 'PRE_YTB_DAV',
    "model": infer_deaot_model(deaot_ckpt),
    "model_path": deaot_ckpt,
    'long_term_mem_gap': 9999,
    'max_len_long_term': 9999,
    "device": "cuda",
}
segtracker_args = {
    'sam_gap': 10, # the interval to run sam to segment new objects
    'min_area': 200, # minimal mask area to add a new mask as a new object
    'max_obj_num': 255, # maximal object number to track in a video
    'min_new_obj_iou': 0.8, # the background area ratio of a new object should > 80%
}