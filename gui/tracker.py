import torch
import gc
from SegTracker import SegTracker
import cv2
from seg_track_anything import draw_mask
import numpy as np
from PIL import Image
from aot_tracker import _palette


def build_putpalette(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    return save_mask


class Tracker:
    def __init__(self, img, start, stop, device, gap=1000, max_len=50) -> None:
        
        torch.cuda.empty_cache()
        gc.collect()
        self.device = device
        print('Tracker:', device, self.device)
        sam_args = {
            'sam_checkpoint': "ckpt/sam_vit_b_01ec64.pth",
            'model_type': "vit_b",
            'generator_args': {
                'points_per_side': 16,
                'pred_iou_thresh': 0.8,
                'stability_score_thresh': 0.9,
                'crop_n_layers': 1,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 200,
            },
            'device': self.device,
        }
        aot_args = {
            'phase': 'PRE_YTB_DAV',
            'model': 'r50_deaotl',
            'model_path': 'ckpt/R50_DeAOTL_PRE_YTB_DAV.pth',
            'long_term_mem_gap': gap,
            'max_len_long_term': max_len,
            'device': self.device,
        }
        segtracker_args = {
            # the interval to run sam to segment new objects
            'sam_gap': int(1e9),
            'min_area': 200,  # minimal mask area to add a new mask as a new object
            'max_obj_num': 10,  # maximal object number to track in a video
            # the background area ratio of a new object should > 80%
            'min_new_obj_iou': 0.8,
        }
        self.first_frame = img
        self.start = start
        self.stop = stop
        self.segtracker = SegTracker(segtracker_args, sam_args, aot_args)
        self.segtracker.restart_tracker()
        self.coords = []
        self.modes = []
        

    def click(self, point, mode):
        self.coords.append(point)
        self.modes.append(mode)

        return self.inference()

    def inference(self):
        if len(self.coords) == 0:
            return None, None
        mask, mix = self.segtracker.seg_acc_click(
            self.first_frame, np.array(self.coords), np.array(self.modes))
        self.mask = mask
        return mask, mix

    def undo(self):
        self.coords = []
        self.modes = []

        return self.inference()

    def next_obj(self):
        if len(self.coords) == 0:
            return
        self.coords = []
        self.modes = []
        self.segtracker.update_origin_merged_mask(self.mask)
        self.segtracker.curr_idx += 1

    def tracking(self, v, progress):
        self.segtracker.add_reference(self.first_frame, self.mask, 0)
        self.segtracker.first_frame_mask = self.mask
        delta = 1 if self.start < self.stop else -1
        print(self.start, self.stop + delta, delta)
        for i in progress.tqdm(range(self.start, self.stop + delta, delta)):
            frame = np.array(Image.open(f'{v.frame_dir}{i:06d}.png'))

            if i == self.start:
                mask = self.segtracker.first_frame_mask
            else:
                mask = self.segtracker.track(frame, update_memory=True)

            mix = draw_mask(frame.copy(), mask)
            mask = build_putpalette(mask).convert(
                mode='RGB')  # type: Image.Image
            mask = np.array(mask)  # type: np.ndarray
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{v.mask_dir}{i:06d}.png', mask)

            mix = cv2.cvtColor(mix, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{v.mix_dir}{i:06d}.png', mix)

        torch.cuda.empty_cache()
        gc.collect()
        self.segtracker.restart_tracker()
        return "Track Finish"
