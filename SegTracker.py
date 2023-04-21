import sys
sys.path.append("..")
sys.path.append("./sam")
from sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from aot_tracker import get_aot
import numpy as np
import torch


class SegTracker():
    def __init__(self,segtracker_args, sam_args,aot_args) -> None:
        """
         Initialize SAM and AOT.
        """
        sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
        sam.to(device=torch.device('cuda',sam_args['gpu_id']))
        self.mask_generator = SamAutomaticMaskGenerator(model=sam,**sam_args['generator_args'])

        self.tracker = get_aot(aot_args)
        self.min_area = segtracker_args['min_area']
        self.max_obj_num = segtracker_args['max_obj_num']
        self.min_new_obj_iou = segtracker_args['min_new_obj_iou']
        self.reference_objs_list = []
    
    def seg(self,frame):
        '''
        Arguments:
            frame: numpy array (h,w,3)
        Return:
            merged_mask: numpy array (h,w)
        '''
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
    
    def add_reference(self,frame,mask,frame_step=0):
        '''
        Add objects in a mask for tracking.
        Arguments:
            frame: numpy array (h,w,3)
            mask: numpy array (h,w)
        '''
        self.reference_objs_list.append(np.unique(mask))
        self.tracker.add_reference_frame(frame,mask,self.get_obj_num(),frame_step)
    
    def track(self,frame,update_memory=False):
        '''
        Track all known objects.
        Arguments:
            frame: numpy array (h,w,3)
        Return:
            merged_mask: numpy array (h,w)
        '''
        pred_mask = self.tracker.track(frame)
        if update_memory:
            self.tracker.update_memory(pred_mask)
        return pred_mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    
    def get_tracking_objs(self):
        objs = set()
        for ref in self.reference_objs_list:
            objs.update(set(ref))
        objs = list(sorted(list(objs)))
        objs = [i for i in objs if i!=0]
        return objs
    
    def get_obj_num(self):
        return int(max(self.get_tracking_objs()))
    
    def find_new_objs(self, track_mask, seg_mask):
        '''
        Compare tracked results from AOT nad segmented results from SAM. Select objects from background if they are not tracked.
        Arguments:
            track_mask: numpy array (h,w)
            seg_mask: numpy array (h,w)
        Return:
            new_obj_mask: numpy array (h,w)
        '''
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

            



