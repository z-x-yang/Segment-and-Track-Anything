import torch
import glob
import os
import numpy as np
import cv2
import PIL

from groundingdino.models import build_model as build_grounding_dino
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T

from torchvision.ops import box_convert


class Detector:
    @staticmethod
    def find_dino_checkpoint():
        base = os.path.dirname(os.path.abspath(__file__))  # folder where detector.py lives
        root = os.path.abspath(os.path.join(base, ".."))  # go up to project root
        folder = os.path.join(root, "ckpt")

        files = sorted(glob.glob(os.path.join(folder, "groundingdino*.pth")))
        if not files:
            raise FileNotFoundError(f"No checkpoint found in {folder}")
        return files[0]

    @staticmethod
    def infer_dino_config(ckpt_path):
        base = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(base, ".."))

        name = os.path.basename(ckpt_path).lower()

        if "swint" in name:
            cfg = "GroundingDINO_SwinT_OGC.py"
        elif "swinb" in name:
            cfg = "GroundingDINO_SwinB_cfg.py"
        else:
            raise ValueError(f"Unknown GroundingDINO model type: {ckpt_path}")

        return os.path.join(root, "groundingdino", "groundingdino", "config", cfg)

    def __init__(self, device):
        grounding_dino_ckpt = self.find_dino_checkpoint()
        config_file = self.infer_dino_config(grounding_dino_ckpt)
        args = SLConfig.fromfile(config_file)
        args.device = device
        self.device = device
        self.gd = build_grounding_dino(args)

        checkpoint = torch.load(grounding_dino_ckpt, map_location='cpu')
        log = self.gd.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(grounding_dino_ckpt, log))
        self.gd.to(device)
        self.gd.eval()

    def image_transform_grounding(self, init_image):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image, _ = transform(init_image, None) # 3, h, w
        return init_image, image

    def image_transform_grounding_for_vis(self, init_image):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
        ])
        image, _ = transform(init_image, None) # 3, h, w
        return image

    def transfer_boxes_format(self, boxes, height, width):
        boxes = boxes * torch.Tensor([width, height, width, height])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

        transfered_boxes = []
        for i in range(len(boxes)):
            box = boxes[i]
            transfered_box = [[int(box[0]), int(box[1])], [int(box[2]), int(box[3])]]
            transfered_boxes.append(transfered_box)

        transfered_boxes = np.array(transfered_boxes)
        return transfered_boxes

    @torch.no_grad()
    def run_grounding(self, origin_frame, grounding_caption, box_threshold, text_threshold):
        '''
            return:
                annotated_frame:nd.array
                transfered_boxes: nd.array [N, 4]: [[x0, y0], [x1, y1]]
        '''
        if isinstance(origin_frame, PIL.Image.Image):
            img_pil = origin_frame.convert("RGB")
            width, height = img_pil.size
        else:
            origin_frame = np.asarray(origin_frame)
            height, width = origin_frame.shape[:2]
            img_pil = PIL.Image.fromarray(origin_frame)
        re_width, re_height = img_pil.size
        _, image_tensor = self.image_transform_grounding(img_pil)
        # img_pil = self.image_transform_grounding_for_vis(img_pil)

        # run grounidng
        boxes, logits, phrases = predict(self.gd, image_tensor, grounding_caption, box_threshold, text_threshold, device=self.device)
        annotated_frame = annotate(image_source=np.asarray(img_pil), boxes=boxes, logits=logits, phrases=phrases)[:, :, ::-1]
        annotated_frame = cv2.resize(annotated_frame, (width, height), interpolation=cv2.INTER_LINEAR)

        # transfer boxes to sam-format
        transfered_boxes = self.transfer_boxes_format(boxes, re_height, re_width)
        return annotated_frame, transfered_boxes

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    origin_frame = cv2.imread('./debug/point.png')
    origin_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
    grounding_caption = "swan.water"
    box_threshold = 0.25
    text_threshold = 0.25
    detector = Detector(device)
    annotated_frame, boxes = detector.run_grounding(origin_frame, grounding_caption, box_threshold, text_threshold)
    cv2.imwrite('./debug/x.png', annotated_frame)

    for i in range(len(boxes)):
        bbox = boxes[i]
        origin_frame = cv2.rectangle(origin_frame, bbox[0], bbox[1], (0, 0, 255))
    cv2.imwrite('./debug/bbox_frame.png', origin_frame)