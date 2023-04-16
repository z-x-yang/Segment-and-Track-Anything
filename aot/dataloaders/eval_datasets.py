from __future__ import division
import os
import shutil
import json
import cv2
from PIL import Image

import numpy as np
from torch.utils.data import Dataset

from utils.image import _palette


class VOSTest(Dataset):
    def __init__(self,
                 image_root,
                 label_root,
                 seq_name,
                 images,
                 labels,
                 rgb=True,
                 transform=None,
                 single_obj=False,
                 resolution=None):
        self.image_root = image_root
        self.label_root = label_root
        self.seq_name = seq_name
        self.images = images
        self.labels = labels
        self.obj_num = 1
        self.num_frame = len(self.images)
        self.transform = transform
        self.rgb = rgb
        self.single_obj = single_obj
        self.resolution = resolution

        self.obj_nums = []
        self.obj_indices = []

        curr_objs = [0]
        for img_name in self.images:
            self.obj_nums.append(len(curr_objs) - 1)
            current_label_name = img_name.split('.')[0] + '.png'
            if current_label_name in self.labels:
                current_label = self.read_label(current_label_name)
                curr_obj = list(np.unique(current_label))
                for obj_idx in curr_obj:
                    if obj_idx not in curr_objs:
                        curr_objs.append(obj_idx)
            self.obj_indices.append(curr_objs.copy())

        self.obj_nums[0] = self.obj_nums[1]

    def __len__(self):
        return len(self.images)

    def read_image(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_root, self.seq_name, img_name)
        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.float32)
        if self.rgb:
            img = img[:, :, [2, 1, 0]]
        return img

    def read_label(self, label_name, squeeze_idx=None):
        label_path = os.path.join(self.label_root, self.seq_name, label_name)
        label = Image.open(label_path)
        label = np.array(label, dtype=np.uint8)
        if self.single_obj:
            label = (label > 0).astype(np.uint8)
        elif squeeze_idx is not None:
            squeezed_label = label * 0
            for idx in range(len(squeeze_idx)):
                obj_id = squeeze_idx[idx]
                if obj_id == 0:
                    continue
                mask = label == obj_id
                squeezed_label += (mask * idx).astype(np.uint8)
            label = squeezed_label
        return label

    def __getitem__(self, idx):
        img_name = self.images[idx]
        current_img = self.read_image(idx)
        height, width, channels = current_img.shape
        if self.resolution is not None:
            width = int(np.ceil(
                float(width) * self.resolution / float(height)))
            height = int(self.resolution)

        current_label_name = img_name.split('.')[0] + '.png'
        obj_num = self.obj_nums[idx]
        obj_idx = self.obj_indices[idx]

        if current_label_name in self.labels:
            current_label = self.read_label(current_label_name, obj_idx)
            sample = {
                'current_img': current_img,
                'current_label': current_label
            }
        else:
            sample = {'current_img': current_img}

        sample['meta'] = {
            'seq_name': self.seq_name,
            'frame_num': self.num_frame,
            'obj_num': obj_num,
            'current_name': img_name,
            'height': height,
            'width': width,
            'flip': False,
            'obj_idx': obj_idx
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class YOUTUBEVOS_Test(object):
    def __init__(self,
                 root='./datasets/YTB',
                 year=2018,
                 split='val',
                 transform=None,
                 rgb=True,
                 result_root=None):
        if split == 'val':
            split = 'valid'
        root = os.path.join(root, str(year), split)
        self.db_root_dir = root
        self.result_root = result_root
        self.rgb = rgb
        self.transform = transform
        self.seq_list_file = os.path.join(self.db_root_dir, 'meta.json')
        self._check_preprocess()
        self.seqs = list(self.ann_f.keys())
        self.image_root = os.path.join(root, 'JPEGImages')
        self.label_root = os.path.join(root, 'Annotations')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_name = self.seqs[idx]
        data = self.ann_f[seq_name]['objects']
        obj_names = list(data.keys())
        images = []
        labels = []
        for obj_n in obj_names:
            images += map(lambda x: x + '.jpg', list(data[obj_n]["frames"]))
            labels.append(data[obj_n]["frames"][0] + '.png')
        images = np.sort(np.unique(images))
        labels = np.sort(np.unique(labels))

        try:
            if not os.path.isfile(
                    os.path.join(self.result_root, seq_name, labels[0])):
                if not os.path.exists(os.path.join(self.result_root,
                                                   seq_name)):
                    os.makedirs(os.path.join(self.result_root, seq_name))
                shutil.copy(
                    os.path.join(self.label_root, seq_name, labels[0]),
                    os.path.join(self.result_root, seq_name, labels[0]))
        except Exception as inst:
            print(inst)
            print('Failed to create a result folder for sequence {}.'.format(
                seq_name))

        seq_dataset = VOSTest(self.image_root,
                              self.label_root,
                              seq_name,
                              images,
                              labels,
                              transform=self.transform,
                              rgb=self.rgb)
        return seq_dataset

    def _check_preprocess(self):
        _seq_list_file = self.seq_list_file
        if not os.path.isfile(_seq_list_file):
            print(_seq_list_file)
            return False
        else:
            self.ann_f = json.load(open(self.seq_list_file, 'r'))['videos']
            return True


class YOUTUBEVOS_DenseTest(object):
    def __init__(self,
                 root='./datasets/YTB',
                 year=2018,
                 split='val',
                 transform=None,
                 rgb=True,
                 result_root=None):
        if split == 'val':
            split = 'valid'
        root_sparse = os.path.join(root, str(year), split)
        root_dense = root_sparse + '_all_frames'
        self.db_root_dir = root_dense
        self.result_root = result_root
        self.rgb = rgb
        self.transform = transform
        self.seq_list_file = os.path.join(root_sparse, 'meta.json')
        self._check_preprocess()
        self.seqs = list(self.ann_f.keys())
        self.image_root = os.path.join(root_dense, 'JPEGImages')
        self.label_root = os.path.join(root_sparse, 'Annotations')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_name = self.seqs[idx]

        data = self.ann_f[seq_name]['objects']
        obj_names = list(data.keys())
        images_sparse = []
        for obj_n in obj_names:
            images_sparse += map(lambda x: x + '.jpg',
                                 list(data[obj_n]["frames"]))
        images_sparse = np.sort(np.unique(images_sparse))

        images = np.sort(
            list(os.listdir(os.path.join(self.image_root, seq_name))))
        start_img = images_sparse[0]
        end_img = images_sparse[-1]
        for start_idx in range(len(images)):
            if start_img in images[start_idx]:
                break
        for end_idx in range(len(images))[::-1]:
            if end_img in images[end_idx]:
                break
        images = images[start_idx:(end_idx + 1)]
        labels = np.sort(
            list(os.listdir(os.path.join(self.label_root, seq_name))))

        try:
            if not os.path.isfile(
                    os.path.join(self.result_root, seq_name, labels[0])):
                if not os.path.exists(os.path.join(self.result_root,
                                                   seq_name)):
                    os.makedirs(os.path.join(self.result_root, seq_name))
                shutil.copy(
                    os.path.join(self.label_root, seq_name, labels[0]),
                    os.path.join(self.result_root, seq_name, labels[0]))
        except Exception as inst:
            print(inst)
            print('Failed to create a result folder for sequence {}.'.format(
                seq_name))

        seq_dataset = VOSTest(self.image_root,
                              self.label_root,
                              seq_name,
                              images,
                              labels,
                              transform=self.transform,
                              rgb=self.rgb)
        seq_dataset.images_sparse = images_sparse

        return seq_dataset

    def _check_preprocess(self):
        _seq_list_file = self.seq_list_file
        if not os.path.isfile(_seq_list_file):
            print(_seq_list_file)
            return False
        else:
            self.ann_f = json.load(open(self.seq_list_file, 'r'))['videos']
            return True


class DAVIS_Test(object):
    def __init__(self,
                 split=['val'],
                 root='./DAVIS',
                 year=2017,
                 transform=None,
                 rgb=True,
                 full_resolution=False,
                 result_root=None):
        self.transform = transform
        self.rgb = rgb
        self.result_root = result_root
        if year == 2016:
            self.single_obj = True
        else:
            self.single_obj = False
        if full_resolution:
            resolution = 'Full-Resolution'
        else:
            resolution = '480p'
        self.image_root = os.path.join(root, 'JPEGImages', resolution)
        self.label_root = os.path.join(root, 'Annotations', resolution)
        seq_names = []
        for spt in split:
            if spt == 'test':
                spt = 'test-dev'
            with open(os.path.join(root, 'ImageSets', str(year),
                                   spt + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            seq_names.extend(seqs_tmp)
        self.seqs = list(np.unique(seq_names))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_name = self.seqs[idx]
        images = list(
            np.sort(os.listdir(os.path.join(self.image_root, seq_name))))
        labels = [images[0].replace('jpg', 'png')]

        if not os.path.isfile(
                os.path.join(self.result_root, seq_name, labels[0])):
            seq_result_folder = os.path.join(self.result_root, seq_name)
            try:
                if not os.path.exists(seq_result_folder):
                    os.makedirs(seq_result_folder)
            except Exception as inst:
                print(inst)
                print(
                    'Failed to create a result folder for sequence {}.'.format(
                        seq_name))
            source_label_path = os.path.join(self.label_root, seq_name,
                                             labels[0])
            result_label_path = os.path.join(self.result_root, seq_name,
                                             labels[0])
            if self.single_obj:
                label = Image.open(source_label_path)
                label = np.array(label, dtype=np.uint8)
                label = (label > 0).astype(np.uint8)
                label = Image.fromarray(label).convert('P')
                label.putpalette(_palette)
                label.save(result_label_path)
            else:
                shutil.copy(source_label_path, result_label_path)

        seq_dataset = VOSTest(self.image_root,
                              self.label_root,
                              seq_name,
                              images,
                              labels,
                              transform=self.transform,
                              rgb=self.rgb,
                              single_obj=self.single_obj,
                              resolution=480)
        return seq_dataset


class _EVAL_TEST(Dataset):
    def __init__(self, transform, seq_name):
        self.seq_name = seq_name
        self.num_frame = 10
        self.transform = transform

    def __len__(self):
        return self.num_frame

    def __getitem__(self, idx):
        current_frame_obj_num = 2
        height = 400
        width = 400
        img_name = 'test{}.jpg'.format(idx)
        current_img = np.zeros((height, width, 3)).astype(np.float32)
        if idx == 0:
            current_label = (current_frame_obj_num * np.ones(
                (height, width))).astype(np.uint8)
            sample = {
                'current_img': current_img,
                'current_label': current_label
            }
        else:
            sample = {'current_img': current_img}

        sample['meta'] = {
            'seq_name': self.seq_name,
            'frame_num': self.num_frame,
            'obj_num': current_frame_obj_num,
            'current_name': img_name,
            'height': height,
            'width': width,
            'flip': False
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class EVAL_TEST(object):
    def __init__(self, transform=None, result_root=None):
        self.transform = transform
        self.result_root = result_root

        self.seqs = ['test1', 'test2', 'test3']

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_name = self.seqs[idx]

        if not os.path.exists(os.path.join(self.result_root, seq_name)):
            os.makedirs(os.path.join(self.result_root, seq_name))

        seq_dataset = _EVAL_TEST(self.transform, seq_name)
        return seq_dataset
