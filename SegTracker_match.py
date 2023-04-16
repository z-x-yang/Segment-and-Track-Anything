import sys
sys.path.append("..")
sys.path.append("./sam")
from sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from aot_tracker import get_aot
import numpy as np
# from scipy.optimize import linear_sum_assignment
import torch


class SegTracker():
    def __init__(self,segtracker_args, sam_args,aot_args) -> None:
        sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
        sam.to(device=torch.device('cuda',sam_args['gpu_id']))
        # self.mask_generator = SamAutomaticMaskGenerator(sam)
        # self.mask_generator = SamAutomaticMaskGenerator(
        #                     model=sam,
        #                     points_per_side=16,
        #                     pred_iou_thresh=0.8,
        #                     stability_score_thresh=0.9,
        #                     crop_n_layers=1,
        #                     crop_n_points_downscale_factor=2,
        #                     min_mask_region_area=200,  # Requires open-cv to run post-processing
        #                 )
        self.mask_generator = SamAutomaticMaskGenerator(model=sam,**sam_args['generator_args'])

        self.tracker = get_aot(aot_args)
        self.match_iou_thr = segtracker_args['match_iou_thr']
        self.min_area = segtracker_args['min_area']
        self.max_obj_num = segtracker_args['max_obj_num']
        self.min_new_obj_iou = segtracker_args['min_new_obj_iou']
        self.reference_objs_list = []
    
    def seg(self,frame):
        anns = self.mask_generator.generate(frame)
        # anns is a list recording all predictions in an image
        if len(anns) == 0:
            return
        # merge all predictions into one mask (h,w)
        # note that the merged mask may lost some objects due to the overlapping
        merged_mask = np.zeros(anns[0]['segmentation'].shape,dtype=np.uint8)
        idx = 1
        for ann in anns:
            if ann['area'] > self.min_area:
                m = ann['segmentation']
                merged_mask[m==1] = idx
                idx += 1
        obj_ids = np.unique(merged_mask)
        obj_ids = obj_ids[obj_ids!=0]
        new_idx = 1
        for id in obj_ids:
            if np.sum(merged_mask==id) < self.min_area or new_idx > self.max_obj_num:
                merged_mask[merged_mask==id] = 0
            else:
                merged_mask[merged_mask==id] = new_idx
                new_idx += 1
        return merged_mask
    
    def add_reference(self,frame,mask):
        self.reference_objs_list.append(np.unique(mask))
        # squeezed_mask = self.squeeze_id(mask)
        self.tracker.add_reference_frame(frame,mask,obj_nums=self.get_obj_num())
    
    def track(self,frame,update_memory=False):
        pred_mask = self.tracker.track(frame)
        if update_memory:
            self.tracker.update_memory(pred_mask)
        # unsqueezed_mask = self.unsqueeze_id(pred_mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8))
        return pred_mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    
    def squeeze_id(self,mask):
        tracking_ids = self.get_tracking_objs()
        squeezed_mask = np.zeros_like(mask)
        squeezed_id = 1
        for id in tracking_ids:
            squeezed_mask[mask==id] = squeezed_id
            squeezed_id += 1
        return squeezed_mask
    
    def unsqueeze_id(self,mask):
        tracking_ids = self.get_tracking_objs()
        unsqueezed_mask = np.zeros_like(mask)
        squeezed_id = 1
        for i in range(len(tracking_ids)):
            unsqueezed_mask[mask==squeezed_id] = tracking_ids[i]
            squeezed_id += 1
        return unsqueezed_mask
    
    def get_tracking_objs(self):
        objs = set()
        for ref in self.reference_objs_list:
            objs.update(set(ref))
        objs = list(sorted(list(objs)))
        objs = [i for i in objs if i!=0]
        return objs
    
    def get_obj_num(self):
        return int(max(self.get_tracking_objs()))
    

    # def match(self, track_mask, seg_mask):
    #     track_ids = np.unique(track_mask)
    #     seg_ids = np.unique(seg_mask)
    #     track_ids = track_ids[track_ids != 0]
    #     seg_ids = seg_ids[seg_ids != 0]
    #     ious = np.zeros((len(track_ids),len(seg_ids)))
    #     for i,track_idx in enumerate(track_ids):
    #         for j,seg_idx in enumerate(seg_ids):
    #             track_obj_mask = track_mask==track_idx
    #             seg_obj_mask = seg_mask==seg_idx

    #             ious[i][j] = np.sum(track_obj_mask & seg_obj_mask) / np.sum(track_obj_mask | seg_obj_mask)
        
    #     match_rows, match_cols = linear_sum_assignment(-ious)
    #     matched_idx = ious[match_rows, match_cols] > self.match_iou_thr
    #     match_rows = match_rows[matched_idx]
    #     match_cols = match_cols[matched_idx]

    #     updated_mask = np.zeros_like(seg_mask)
    #     # reset ids in seg mask by track ids
    #     for i,j in zip(match_rows,match_cols):
    #         updated_mask[seg_mask==seg_ids[j]] = track_ids[i]
    #     # add new ids in seg mask
    #     new_obj_id = self.get_obj_num() + 1
    #     for col_idx in range(len(seg_ids)):
    #         if col_idx not in match_cols:
    #             updated_mask[seg_mask==seg_ids[col_idx]] = new_obj_id
    #             new_obj_id += 1
        
    #     return updated_mask
    
    def find_new_objs(self, track_mask, seg_mask):
        new_obj_mask = (track_mask==0) * seg_mask
        new_obj_ids = np.unique(new_obj_mask)
        new_obj_ids = new_obj_ids[new_obj_ids!=0]
        obj_num = self.get_obj_num() + 1
        for idx in new_obj_ids:
            new_obj_area = np.sum(new_obj_mask==idx)
            obj_area = np.sum(seg_mask==idx)
            if new_obj_area/obj_area < self.min_new_obj_iou or new_obj_area < self.min_area\
                or obj_num > self.max_obj_num:
                new_obj_mask[new_obj_mask==idx] = 0
            else:
                new_obj_mask[new_obj_mask==idx] = obj_num
                obj_num += 1
        return new_obj_mask
        
        
    def restart_tracker(self):
        self.tracker.restart()

            



