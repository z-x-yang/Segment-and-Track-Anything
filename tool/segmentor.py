import time
import torch
import cv2
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from typing import Union
from sam.segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import PIL
from .mask_painter import mask_painter
from .painter import  point_painter

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


class Segmentor:
    def __init__(self, sam_args):
        """
        sam_args:
            model_type: vit_b, vit_l, vit_h
            sam_checkpoint: path of SAM checkpoint
            generator_args: args for everything_generator
            gpu_id: device
        """
        print(f"Initializing Segmentor to {sam_args['gpu_id']}")
        assert sam_args["model_type"] in ['vit_b', 'vit_l', 'vit_h'], 'model_type must be vit_b, vit_l, or vit_h'

        self.device = sam_args["gpu_id"]
        # self.torch_dtype = torch.float16 if 'cuda' in sam_args["gpu_id"] else torch.float32
        self.model = sam_model_registry[sam_args["model_type"]](checkpoint=sam_args["sam_checkpoint"])
        self.model.to(device=self.device)
        self.everything_generator = SamAutomaticMaskGenerator(model=self.model,**sam_args['generator_args'])
        self.interactive_predictor = self.everything_generator.predictor
        self.embedded = False

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        # PIL.open(image_path) 3channel: RGB
        # image embedding: avoid encode the same image multiple times

        self.orignal_image = image
        if self.embedded:
            print('repeat embedding, please reset_image.')
            return
        self.interactive_predictor.set_image(image)
        self.embedded = True
        return
    
    @torch.no_grad()
    def reset_image(self):
        # reset image embeding
        self.interactive_predictor.reset_image()
        self.embedded = False

    def interactive_predict(self, prompts, mode, multimask=True):
        """
        image: numpy array, h, w, 3
        prompts: dictionary, 3 keys: 'point_coords', 'point_labels', 'mask_input'
        prompts['point_coords']: numpy array [N,2]
        prompts['point_labels']: numpy array [1,N]
        prompts['mask_input']: numpy array [1,256,256]
        mode: 'point' (points only), 'mask' (mask only), 'both' (consider both)
        mask_outputs: True (return 3 masks), False (return 1 mask only)
        whem mask_outputs=True, mask_input=logits[np.argmax(scores), :, :][None, :, :]
        """
        assert self.embedded, 'prediction is called before set_image (feature embedding).'
        assert mode in ['point', 'mask', 'both'], 'mode must be point, mask, or both'
        
        if mode == 'point':
            masks, scores, logits = self.interactive_predictor.predict(point_coords=prompts['point_coords'], 
                                point_labels=prompts['point_labels'], 
                                multimask_output=multimask)
        elif mode == 'mask':
            masks, scores, logits = self.interactive_predictor.predict(mask_input=prompts['mask_input'], 
                                multimask_output=multimask)
        elif mode == 'both':   # both
            masks, scores, logits = self.interactive_predictor.predict(point_coords=prompts['point_coords'], 
                                point_labels=prompts['point_labels'], 
                                mask_input=prompts['mask_input'], 
                                multimask_output=multimask)
        else:
            raise("Not implement now!")
        # masks (n, h, w), scores (n,), logits (n, 256, 256)
        return masks, scores, logits

    
    def segment_with_click(self, origin_frame: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        '''
            return: 
                mask: one-hot 
                logit:
                painted_iamge: paint mask and point
        '''
        self.set_image(origin_frame)

        neg_flag = labels[-1]   # 1 indicates a foreground point and 0 indicates a background point.
        # if neg_flag == 1:
        #find positive
        prompts = {
            'point_coords': points,
            'point_labels': labels,
        }
        masks, scores, logits = self.interactive_predict(prompts, 'point', multimask)
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        prompts = {
            'point_coords': points,
            'point_labels': labels,
            'mask_input': logit[None, :, :]
        }
        masks, scores, logits = self.interactive_predict(prompts, 'both', multimask)
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
            
        # else:
        #     #find neg
        #     prompts = {
        #         'point_coords': points,
        #         'point_labels': labels,
        #     }
        #     masks, scores, logits = self.interactive_predict(prompts, 'point', multimask)
        #     mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]

        assert len(points)==len(labels)
        outline = mask_painter(origin_frame.copy(), mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)
        # painted_image = Image.fromarray(painted_image)
        return mask.astype(np.uint8), logit, outline

    def segment_with_box(self, origin_frame, bbox):
        self.set_image(origin_frame)

        masks , _, _ = self.interactive_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([[bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]]),
            multimask_output=False
        )
        
        return masks

# if __name__ == "__main__":
#     points = np.array([[500, 375], [1125, 625]])
#     labels = np.array([1, 1])
#     image = cv2.imread('/hhd3/gaoshang/truck.jpg')
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     sam_controler = initialize()
#     mask, logit, painted_image_full = first_frame_click(sam_controler,image, points, labels, multimask=True)
#     painted_image = mask_painter2(image, mask.astype('uint8'), background_alpha=0.8)
#     painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
#     cv2.imwrite('/hhd3/gaoshang/truck_point.jpg', painted_image)
#     cv2.imwrite('/hhd3/gaoshang/truck_change.jpg', image)
#     painted_image_full.save('/hhd3/gaoshang/truck_point_full.jpg')
    
#     mask, logit, painted_image_full = interact_loop(sam_controler,image,True, points, np.array([1, 0]), logit, multimask=True)
#     painted_image = mask_painter2(image, mask.astype('uint8'), background_alpha=0.8)
#     painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
#     cv2.imwrite('/hhd3/gaoshang/truck_same.jpg', painted_image)
#     painted_image_full.save('/hhd3/gaoshang/truck_same_full.jpg')
    
#     mask, logit, painted_image_full = interact_loop(sam_controler,image, False, points, labels, multimask=True)
#     painted_image = mask_painter2(image, mask.astype('uint8'), background_alpha=0.8)
#     painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
#     cv2.imwrite('/hhd3/gaoshang/truck_diff.jpg', painted_image)
#     painted_image_full.save('/hhd3/gaoshang/truck_diff_full.jpg')