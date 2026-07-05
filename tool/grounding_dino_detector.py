import os
import cv2
import numpy as np
import PIL
import torch
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

class Detector:
    MODEL_ID = "IDEA-Research/grounding-dino-base"

    def __init__(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                print("CUDA not available. Falling back to MPS.")
                device = "mps"
            else:
                print("CUDA not available. Falling back to CPU.")
                device = "cpu"

        self.device = device
        print(f"Loading GroundingDINO model: {self.MODEL_ID}")

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.MODEL_ID
        ).to(self.device)

        self.model.eval()

    @staticmethod
    def normalize_caption(text: str):
        """
        HuggingFace GroundingDINO expects:
            - lowercase
            - each phrase separated by '.'
            - final '.'
        """
        parts = [
            part.strip().lower()
            for part in text.split(".")
            if part.strip()
        ]

        return ". ".join(parts) + "."

    @torch.no_grad()
    def run_grounding(
        self,
        origin_frame,
        grounding_caption,
        box_threshold,
        text_threshold,
    ):
        """
        Returns:
            annotated_frame : np.ndarray
            transfered_boxes : np.ndarray

            Shape:
            [
                [[x0, y0], [x1, y1]],
                ...
            ]
        """

        if isinstance(origin_frame, PIL.Image.Image):
            img_pil = origin_frame.convert("RGB")
        else:
            origin_frame = np.asarray(origin_frame)
            img_pil = PIL.Image.fromarray(origin_frame)

        grounding_caption = self.normalize_caption(grounding_caption)

        inputs = self.processor(
            images=img_pil,
            text=grounding_caption,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[img_pil.size[::-1]],
        )[0]

        annotated_frame = origin_frame.copy()

        transfered_boxes = []

        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"],
        ):
            x1, y1, x2, y2 = map(int, box.tolist())

            transfered_boxes.append(
                [
                    [x1, y1],
                    [x2, y2],
                ]
            )

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"{label}: {score:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        transfered_boxes = np.asarray(
            transfered_boxes,
            dtype=np.int32,
        )

        return annotated_frame, transfered_boxes


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detector = Detector(device)

    origin_frame = cv2.imread("./tutorial/img/click_segment_everything.jpg")
    origin_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)

    grounding_caption = "swan.water"

    box_threshold = 0.25
    text_threshold = 0.25

    annotated_frame, boxes = detector.run_grounding(
        origin_frame,
        grounding_caption,
        box_threshold,
        text_threshold,
    )

    cv2.imwrite(
        "./debug/x.png",
        cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR),
    )

    bbox_frame = origin_frame.copy()

    for bbox in boxes:
        cv2.rectangle(
            bbox_frame,
            tuple(bbox[0]),
            tuple(bbox[1]),
            (0, 0, 255),
            2,
        )

    cv2.imwrite(
        "./debug/bbox_frame.png",
        cv2.cvtColor(bbox_frame, cv2.COLOR_RGB2BGR),
    )