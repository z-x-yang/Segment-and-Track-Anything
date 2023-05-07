from statistics import mode
import torch
import torch.nn.functional as F
import os
import sys
sys.path.append("./aot")
from aot.networks.engines.aot_engine import AOTEngine,AOTInferEngine
from aot.networks.engines.deaot_engine import DeAOTEngine,DeAOTInferEngine
import importlib
import numpy as np
from PIL import Image
from skimage.morphology.binary import binary_dilation


np.random.seed(200)
_palette = ((np.random.random((3*255))*0.7+0.3)*255).astype(np.uint8).tolist()
_palette = [0,0,0]+_palette

import aot.dataloaders.video_transforms as tr
from aot.utils.checkpoint import load_network
from aot.networks.models import build_vos_model
from aot.networks.engines import build_engine
from torchvision import transforms

class AOTTracker(object):
    def __init__(self, cfg, gpu_id=0):
        self.gpu_id = gpu_id
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        # self.engine = self.build_tracker_engine(cfg.MODEL_ENGINE,
        #                            aot_model=self.model,
        #                            gpu_id=gpu_id,
        #                            short_term_mem_skip=4,
        #                            long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)
        self.engine = build_engine(cfg.MODEL_ENGINE,
                                   phase='eval',
                                   aot_model=self.model,
                                   gpu_id=gpu_id,
                                   short_term_mem_skip=1,
                                   long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP,
                                   max_len_long_term=cfg.MAX_LEN_LONG_TERM)
       
        self.transform = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MAX_SHORT_EDGE,
                                 cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP, 
                                 cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])

        self.model.eval()

    @torch.no_grad()
    def add_reference_frame(self, frame, mask, obj_nums, frame_step, incremental=False):
        # mask = cv2.resize(mask, frame.shape[:2][::-1], interpolation = cv2.INTER_NEAREST)

        sample = {
            'current_img': frame,
            'current_label': mask,
        }
    
        sample = self.transform(sample)
        frame = sample[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id)
        mask = sample[0]['current_label'].unsqueeze(0).float().cuda(self.gpu_id)
        _mask = F.interpolate(mask,size=frame.shape[-2:],mode='nearest')

        if incremental:
            self.engine.add_reference_frame_incremental(frame, _mask, obj_nums=obj_nums, frame_step=frame_step)
        else:
            self.engine.add_reference_frame(frame, _mask, obj_nums=obj_nums, frame_step=frame_step)



    @torch.no_grad()
    def track(self, image):
        output_height, output_width = image.shape[0], image.shape[1]
        sample = {'current_img': image}
        sample = self.transform(sample)
        image = sample[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id)
        self.engine.match_propogate_one_frame(image)
        pred_logit = self.engine.decode_current_logits((output_height, output_width))

        # pred_prob = torch.softmax(pred_logit, dim=1)
        pred_label = torch.argmax(pred_logit, dim=1,
                                    keepdim=True).float()

        return  pred_label
    
    @torch.no_grad()
    def update_memory(self, pred_label):
        self.engine.update_memory(pred_label)
    
    @torch.no_grad()
    def restart(self):
        self.engine.restart_engine()
    
    @torch.no_grad()
    def build_tracker_engine(self, name, **kwargs):
        if name == 'aotengine':
            return AOTTrackerInferEngine(**kwargs)
        elif name == 'deaotengine':
            return DeAOTTrackerInferEngine(**kwargs)
        else:
            raise NotImplementedError


class AOTTrackerInferEngine(AOTInferEngine):
    def __init__(self, aot_model, gpu_id=0, long_term_mem_gap=9999, short_term_mem_skip=1, max_aot_obj_num=None):
        super().__init__(aot_model, gpu_id, long_term_mem_gap, short_term_mem_skip, max_aot_obj_num)
    def add_reference_frame_incremental(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while (aot_num > len(self.aot_engines)):
            new_engine = AOTEngine(self.AOT, self.gpu_id,
                                   self.long_term_mem_gap,
                                   self.short_term_mem_skip)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(
            mask, obj_nums)
        img_embs = None
        for aot_engine, separated_mask, separated_obj_num in zip(
                self.aot_engines, separated_masks, separated_obj_nums):
            if aot_engine.obj_nums is None or aot_engine.obj_nums[0] < separated_obj_num:
                aot_engine.add_reference_frame(img,
                                            separated_mask,
                                            obj_nums=[separated_obj_num],
                                            frame_step=frame_step,
                                            img_embs=img_embs)
            else:
                aot_engine.update_short_term_memory(separated_mask)
                
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

        self.update_size()



class DeAOTTrackerInferEngine(DeAOTInferEngine):
    def __init__(self, aot_model, gpu_id=0, long_term_mem_gap=9999, short_term_mem_skip=1, max_aot_obj_num=None):
        super().__init__(aot_model, gpu_id, long_term_mem_gap, short_term_mem_skip, max_aot_obj_num)
    def add_reference_frame_incremental(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while (aot_num > len(self.aot_engines)):
            new_engine = DeAOTEngine(self.AOT, self.gpu_id,
                                   self.long_term_mem_gap,
                                   self.short_term_mem_skip)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(
            mask, obj_nums)
        img_embs = None
        for aot_engine, separated_mask, separated_obj_num in zip(
                self.aot_engines, separated_masks, separated_obj_nums):
            if aot_engine.obj_nums is None or aot_engine.obj_nums[0] < separated_obj_num:
                aot_engine.add_reference_frame(img,
                                            separated_mask,
                                            obj_nums=[separated_obj_num],
                                            frame_step=frame_step,
                                            img_embs=img_embs)
            else:
                aot_engine.update_short_term_memory(separated_mask)
                
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

        self.update_size()


def get_aot(args):
    # build vos engine
    engine_config = importlib.import_module('configs.' + 'pre_ytb_dav')
    cfg = engine_config.EngineConfig(args['phase'], args['model'])
    cfg.TEST_CKPT_PATH = args['model_path']
    cfg.TEST_LONG_TERM_MEM_GAP = args['long_term_mem_gap']
    cfg.MAX_LEN_LONG_TERM = args['max_len_long_term']
    # init AOTTracker
    tracker = AOTTracker(cfg, args['gpu_id'])
    return tracker
