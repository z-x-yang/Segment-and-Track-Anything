import torch
import gc
import datetime
import os
import cv2
import glob
import numpy as np
from PIL import Image

class Video_obj:
    def __init__(self, file, progress) -> None:
        torch.cuda.empty_cache()
        gc.collect()
        self.video = file.name
        self.build_folder()
        self.read_video(progress)

    def build_folder(self) -> None:
        now = datetime.datetime.today()
        now = str(now).split('.')[0].replace(':', '-').replace(' ', '-')
        basename = os.path.basename(self.video).split('.')[0]
        self.folder = f'result/{now}-{basename}/'
        self.frame_dir = f'{self.folder}frame/'
        self.mask_dir = f'{self.folder}mask/'
        self.mix_dir = f'{self.folder}mix/'
        self.obj_li = []
        for it in [self.folder, self.frame_dir, self.mask_dir, self.mix_dir]:
            os.makedirs(it, exist_ok=True)

    def read_video(self, progress):
        cap = cv2.VideoCapture(self.video)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in progress.tqdm(range(self.n_frame)):
            ok, frame = cap.read()
            assert ok, "[E] Read Frame Error"

            name = f'{self.frame_dir}{i:06d}.png'
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(name, frame)

        cap.release()

    def write_video(self, mode):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        basename = os.path.basename(self.video).split('.')[0]
        name = f'{self.folder}{basename}{mode}.mp4'
        video = cv2.VideoWriter(name, fourcc, self.fps,
                                (self.width, self.height))
        if mode == 'mask':
            li = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))
        else:
            li = sorted(glob.glob(os.path.join(self.mix_dir, '*.png')))

        for img_file in li:
            img = cv2.imread(img_file)
            video.write(img)

        video.release()

        return name

    def add_object(self, name):
        self.obj_li.append(name)
        return len(self.obj_li)

    def display(self, i, mode):
        if i < 0 or i >= self.n_frame:
            return None

        if mode == 'Image':
            try:
                return np.array(Image.open(f'{self.frame_dir}{i:06d}.png'))
            except:
                return None

        if mode == 'Image & Mask':
            try:
                return np.array(Image.open(f'{self.mix_dir}{i:06d}.png'))
            except:
                return None

        if mode == 'Mask':
            try:
                return np.array(Image.open(f'{self.mask_dir}{i:06d}.png'))
            except:
                return None

