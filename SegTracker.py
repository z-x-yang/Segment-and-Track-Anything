import sys
sys.path.append("..")
sys.path.append("./sam")
from sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from aot_tracker import get_aot
import numpy as np
import torch
from tool.segmentor import Segmentor
import cv2
import os
from PIL import Image
import gc
from aot_tracker import _palette
from seg_track_anything import draw_mask
from tool.painter import  point_painter

mask_color = 3
mask_alpha = 0.7
contour_color = 1
contour_width = 5
point_color_ne = 8
point_color_ps = 50
point_alpha = 0.9
point_radius = 15
contour_color = 2
contour_width = 5


class SegTracker():
    def __init__(self,segtracker_args, sam_args,aot_args) -> None:
        """
         Initialize SAM and AOT.
        """
        self.sam = Segmentor(sam_args)
        self.tracker = get_aot(aot_args)
        self.sam_gap = segtracker_args['sam_gap']
        self.min_area = segtracker_args['min_area']
        self.max_obj_num = segtracker_args['max_obj_num']
        self.min_new_obj_iou = segtracker_args['min_new_obj_iou']
        self.reference_objs_list = []
        self.object_idx = 1
        self.origin_merged_mask = None  # init with 0 or segment-everthing
        self.refined_merged_mask = None # interactively refine by user

        # debug
        self.everything_points = []
        self.everything_labels = []

    def seg(self,frame):
        '''
        Arguments:
            frame: numpy array (h,w,3)
        Return:
            origin_merged_mask: numpy array (h,w)
        '''
        frame = frame[:, :, ::-1]
        anns = self.sam.everything_generator.generate(frame)

        # anns is a list recording all predictions in an image
        if len(anns) == 0:
            return
        # merge all predictions into one mask (h,w)
        # note that the merged mask may lost some objects due to the overlapping
        self.origin_merged_mask = np.zeros(anns[0]['segmentation'].shape,dtype=np.uint8)
        idx = 1
        for ann in anns:
            if ann['area'] > self.min_area:
                m = ann['segmentation']
                self.origin_merged_mask[m==1] = idx
                idx += 1
                self.everything_points.append(ann["point_coords"][0])
                self.everything_labels.append(1)

        obj_ids = np.unique(self.origin_merged_mask)
        obj_ids = obj_ids[obj_ids!=0]

        self.object_idx = 1
        for id in obj_ids:
            if np.sum(self.origin_merged_mask==id) < self.min_area or self.object_idx > self.max_obj_num:
                self.origin_merged_mask[self.origin_merged_mask==id] = 0
            else:
                self.origin_merged_mask[self.origin_merged_mask==id] = self.object_idx
                self.object_idx += 1

        self.refined_merged_mask = self.origin_merged_mask
        return self.origin_merged_mask
    
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
            origin_merged_mask: numpy array (h,w)
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

    def seg_acc_bbox(self, origin_frame: np.ndarray, bbox: np.ndarray,):
        ''''
        parameters:
            origin_frame: H, W, C
            bbox: [[x0, y0], [x1, y1]]
        '''

        # get interactive_mask
        interactive_mask = self.sam.segment_with_box(origin_frame, bbox)[0]
        self.refined_merged_mask = self.add_mask(interactive_mask)

        # draw mask
        masked_frame = draw_mask(origin_frame.copy(), self.refined_merged_mask)

        # draw bbox
        masked_frame = cv2.rectangle(masked_frame, bbox[0], bbox[1], (0, 0, 255))

        return self.refined_merged_mask, masked_frame

    def refine_first_frame_click(self, origin_frame: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        '''
        it is used in first frame in video
        return: mask, logit, painted image(mask+point)
        '''
        # get interactive_mask
        interactive_mask, logit, outline = self.sam.segment_with_click(origin_frame, points, labels, multimask)

        self.refined_merged_mask = self.add_mask(interactive_mask)

        # draw mask
        masked_frame = draw_mask(origin_frame.copy(), self.refined_merged_mask)

        # draw points
        # self.everything_labels = np.array(self.everything_labels).astype(np.int64)
        # self.everything_points = np.array(self.everything_points).astype(np.int64)
        # masked_frame = point_painter(masked_frame, np.squeeze(self.everything_points[np.argwhere(self.everything_labels==1)], axis = 1), point_color_ps, point_alpha, point_radius, contour_color, contour_width)

        masked_frame = point_painter(masked_frame, np.squeeze(points[np.argwhere(labels==0)], axis = 1), point_color_ne, point_alpha, point_radius, contour_color, contour_width)
        masked_frame = point_painter(masked_frame, np.squeeze(points[np.argwhere(labels==1)], axis = 1), point_color_ps, point_alpha, point_radius, contour_color, contour_width)
        # draw outline
        masked_frame = np.where(outline > 0, outline, masked_frame)

        return self.refined_merged_mask, masked_frame

    def add_mask(self, interactive_mask, cover_origin_objects=True, single_object=True):
        # if cover_origin_objects == Ture: interactive_mask will cover original object
        # if single_object == True: added mask is belong to single object
        if not cover_origin_objects:
            empty_mask = np.where(self.origin_merged_mask == 0, 1, 0)
            interactive_mask = interactive_mask * empty_mask
        
        if self.origin_merged_mask is None:
            self.origin_merged_mask = np.zeros(interactive_mask.shape,dtype=np.uint8)

        refined_merged_mask = self.origin_merged_mask.copy()
        refined_merged_mask[interactive_mask > 0] = self.object_idx

        if not single_object:
            self.object_idx += 1

        return refined_merged_mask
    

if __name__ == '__main__':
    from model_args import segtracker_args,sam_args,aot_args

    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    # Seg_Tracker.restart_tracker()

    origin_frame = cv2.imread('/data2/cym/Seg_Tra_any/Segment-and-Track-Anything/debug/point.png')
    origin_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)

    merged_mask = Seg_Tracker.seg(origin_frame)
    cv2.imwrite('./debug/merged_mask.png', -1)

    # one positive point
    # point = np.array([[300, 420]])
    # label = np.array([0])

    # two positive point
    point = np.array([[250, 370], [300, 420], [480, 150]])
    label = np.array([1, 0, 1])

    prompt = {
        "prompt_type":["click"],
        "input_point":point,
        "input_label":label,
        "multimask_output":"True",
    }

    predicted_mask, masked_frame = Seg_Tracker.refine_first_frame_click( 
        origin_frame=origin_frame, 
        points=np.array(prompt["input_point"]),
        labels=np.array(prompt["input_label"]),
        multimask=prompt["multimask_output"],
    )
    
    masked_frame = Image.fromarray(masked_frame)
    masked_frame.save('./debug/masked_frame.png')
