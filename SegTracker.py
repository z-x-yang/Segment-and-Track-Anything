import sys
sys.path.append("..")
sys.path.append("./sam")
from sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from aot_tracker import get_aot
import numpy as np
from tool.segmentor import Segmentor
from tool.detector import Detector
from tool.transfer_tools import draw_outline, draw_points
import cv2
from seg_track_anything import draw_mask


class SegTracker():
    def __init__(self,segtracker_args, sam_args, aot_args) -> None:
        """
         Initialize SAM and AOT.
        """
        self.sam = Segmentor(sam_args)
        self.tracker = get_aot(aot_args)
        self.detector = Detector(self.sam.device)
        self.sam_gap = segtracker_args['sam_gap']
        self.min_area = segtracker_args['min_area']
        self.max_obj_num = segtracker_args['max_obj_num']
        self.min_new_obj_iou = segtracker_args['min_new_obj_iou']
        self.reference_objs_list = []
        self.object_idx = 1
        self.curr_idx = 1
        self.origin_merged_mask = None  # init by segment-everything or update
        self.first_frame_mask = None

        # debug
        self.everything_points = []
        self.everything_labels = []
        print("SegTracker has been initialized")

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

        self.first_frame_mask = self.origin_merged_mask
        return self.origin_merged_mask

    def update_origin_merged_mask(self, updated_merged_mask):
        self.origin_merged_mask = updated_merged_mask
        # obj_ids = np.unique(updated_merged_mask)
        # obj_ids = obj_ids[obj_ids!=0]
        # self.object_idx = int(max(obj_ids)) + 1

    def reset_origin_merged_mask(self, mask, id):
        self.origin_merged_mask = mask
        self.curr_idx = id

    def add_reference(self,frame,mask,frame_step=0):
        '''
        Add objects in a mask for tracking.
        Arguments:
            frame: numpy array (h,w,3)
            mask: numpy array (h,w)
        '''
        self.reference_objs_list.append(np.unique(mask))
        self.curr_idx = self.get_obj_num()
        self.tracker.add_reference_frame(frame,mask, self.curr_idx, frame_step)

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
        objs = self.get_tracking_objs()
        if len(objs) == 0: return 0
        return int(max(objs))

    def find_new_objs(self, track_mask, seg_mask):
        '''
        Compare tracked results from AOT with segmented results from SAM. Select objects from background if they are not tracked.
        Arguments:
            track_mask: numpy array (h,w)
            seg_mask: numpy array (h,w)
        Return:
            new_obj_mask: numpy array (h,w)
        '''
        new_obj_mask = (track_mask==0) * seg_mask
        new_obj_ids = np.unique(new_obj_mask)
        new_obj_ids = new_obj_ids[new_obj_ids!=0]
        # obj_num = self.get_obj_num() + 1
        obj_num = self.curr_idx
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
        Use bbox-prompt to get mask
        Parameters:
            origin_frame: H, W, C
            bbox: [[x0, y0], [x1, y1]]
        Return:
            refined_merged_mask: numpy array (h, w)
            masked_frame: numpy array (h, w, c)
        '''
        # get interactive_mask
        interactive_mask = self.sam.segment_with_box(origin_frame, bbox)[0]
        refined_merged_mask = self.add_mask(interactive_mask)

        # draw mask
        masked_frame = draw_mask(origin_frame.copy(), refined_merged_mask)

        # draw bbox
        masked_frame = cv2.rectangle(masked_frame, bbox[0], bbox[1], (0, 0, 255))

        return refined_merged_mask, masked_frame

    def seg_acc_click(self, origin_frame: np.ndarray, coords: np.ndarray, modes: np.ndarray, multimask=True):
        '''
        Use point-prompt to get mask
        Parameters:
            origin_frame: H, W, C
            coords: nd.array [[x, y]]
            modes: nd.array [[1]]
        Return:
            refined_merged_mask: numpy array (h, w)
            masked_frame: numpy array (h, w, c)
        '''
        # get interactive_mask
        interactive_mask = self.sam.segment_with_click(origin_frame, coords, modes, multimask)

        refined_merged_mask = self.add_mask(interactive_mask)

        # draw mask
        masked_frame = draw_mask(origin_frame.copy(), refined_merged_mask)

        # draw points
        # self.everything_labels = np.array(self.everything_labels).astype(np.int64)
        # self.everything_points = np.array(self.everything_points).astype(np.int64)

        masked_frame = draw_points(coords, modes, masked_frame)

        # draw outline
        masked_frame = draw_outline(interactive_mask, masked_frame)

        return refined_merged_mask, masked_frame

    def add_mask(self, interactive_mask: np.ndarray):
        '''
        Merge interactive mask with self.origin_merged_mask
        Parameters:
            interactive_mask: numpy array (h, w)
        Return:
            refined_merged_mask: numpy array (h, w)
        '''
        if self.origin_merged_mask is None:
            self.origin_merged_mask = np.zeros(interactive_mask.shape,dtype=np.uint8)

        refined_merged_mask = self.origin_merged_mask.copy()
        refined_merged_mask[interactive_mask > 0] = self.curr_idx

        return refined_merged_mask
    
    def detect_and_seg(self, origin_frame: np.ndarray, grounding_caption, box_threshold, text_threshold, box_size_threshold=1, reset_image=False):
        '''
        Using Grounding-DINO to detect object acc Text-prompts
        Retrun:
            refined_merged_mask: numpy array (h, w)
            annotated_frame: numpy array (h, w, 3)
        '''
        # backup id and origin-merged-mask
        bc_id = self.curr_idx
        bc_mask = self.origin_merged_mask

        # get annotated_frame and boxes
        annotated_frame, boxes = self.detector.run_grounding(origin_frame, grounding_caption, box_threshold, text_threshold)
        for i in range(len(boxes)):
            bbox = boxes[i]
            if (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1]) > annotated_frame.shape[0] * annotated_frame.shape[1] * box_size_threshold:
                continue
            interactive_mask = self.sam.segment_with_box(origin_frame, bbox, reset_image)[0]
            refined_merged_mask = self.add_mask(interactive_mask)
            self.update_origin_merged_mask(refined_merged_mask)
            self.curr_idx += 1

        # reset origin_mask
        self.reset_origin_merged_mask(bc_mask, bc_id)

        return refined_merged_mask, annotated_frame

if __name__ == '__main__':
    from model_args import segtracker_args,sam_args,aot_args

    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    
    # ------------------ detect test ----------------------
    
    origin_frame = cv2.imread('/data2/cym/Seg_Tra_any/Segment-and-Track-Anything/debug/point.png')
    origin_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
    grounding_caption = "swan.water"
    box_threshold = 0.25
    text_threshold = 0.25

    predicted_mask, annotated_frame = Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold, text_threshold)
    masked_frame = draw_mask(annotated_frame, predicted_mask)
    origin_frame = cv2.cvtColor(origin_frame, cv2.COLOR_RGB2BGR)

    cv2.imwrite('./debug/masked_frame.png', masked_frame)
    cv2.imwrite('./debug/x.png', annotated_frame)