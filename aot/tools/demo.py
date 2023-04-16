import importlib
import sys
import os

sys.path.append('.')
sys.path.append('..')

import cv2
from PIL import Image
from skimage.morphology.binary import binary_dilation

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from networks.models import build_vos_model
from networks.engines import build_engine
from utils.checkpoint import load_network

from dataloaders.eval_datasets import VOSTest
import dataloaders.video_transforms as tr
from utils.image import save_mask

_palette = [
    255, 0, 0, 0, 0, 139, 255, 255, 84, 0, 255, 0, 139, 0, 139, 0, 128, 128,
    128, 128, 128, 139, 0, 0, 218, 165, 32, 144, 238, 144, 160, 82, 45, 148, 0,
    211, 255, 0, 255, 30, 144, 255, 255, 218, 185, 85, 107, 47, 255, 140, 0,
    50, 205, 50, 123, 104, 238, 240, 230, 140, 72, 61, 139, 128, 128, 0, 0, 0,
    205, 221, 160, 221, 143, 188, 143, 127, 255, 212, 176, 224, 230, 244, 164,
    96, 250, 128, 114, 70, 130, 180, 0, 128, 0, 173, 255, 47, 255, 105, 180,
    238, 130, 238, 154, 205, 50, 220, 20, 60, 176, 48, 96, 0, 206, 209, 0, 191,
    255, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45,
    45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51,
    52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58,
    58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64,
    64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70,
    71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77,
    77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83,
    83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89,
    90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96,
    96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101,
    102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106,
    107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111,
    112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116,
    117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121,
    122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126,
    127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131,
    132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136,
    137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141,
    142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146,
    147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151,
    152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156,
    157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161,
    162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166,
    167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171,
    172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176,
    177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181,
    182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186,
    187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191,
    192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196,
    197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201,
    202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206,
    207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211,
    212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216,
    217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221,
    222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226,
    227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231,
    232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236,
    237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241,
    242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246,
    247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251,
    252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255, 0, 0, 0
]
color_palette = np.array(_palette).reshape(-1, 3)


def overlay(image, mask, colors=[255, 0, 0], cscale=1, alpha=0.4):
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask

        foreground = image * alpha + np.ones(
            image.shape) * (1 - alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        countours = binary_dilation(binary_mask) ^ binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


def demo(cfg):
    video_fps = 15
    gpu_id = cfg.TEST_GPU_ID

    # Load pre-trained model
    print('Build AOT model.')
    model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)

    print('Load checkpoint from {}'.format(cfg.TEST_CKPT_PATH))
    model, _ = load_network(model, cfg.TEST_CKPT_PATH, gpu_id)

    print('Build AOT engine.')
    engine = build_engine(cfg.MODEL_ENGINE,
                          phase='eval',
                          aot_model=model,
                          gpu_id=gpu_id,
                          long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)

    # Prepare datasets for each sequence
    transform = transforms.Compose([
        tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE,
                             cfg.TEST_FLIP, cfg.TEST_MULTISCALE,
                             cfg.MODEL_ALIGN_CORNERS),
        tr.MultiToTensor()
    ])
    image_root = os.path.join(cfg.TEST_DATA_PATH, 'images')
    label_root = os.path.join(cfg.TEST_DATA_PATH, 'masks')

    sequences = os.listdir(image_root)
    seq_datasets = []
    for seq_name in sequences:
        print('Build a dataset for sequence {}.'.format(seq_name))
        seq_images = np.sort(os.listdir(os.path.join(image_root, seq_name)))
        seq_labels = [seq_images[0].replace('jpg', 'png')]
        seq_dataset = VOSTest(image_root,
                              label_root,
                              seq_name,
                              seq_images,
                              seq_labels,
                              transform=transform)
        seq_datasets.append(seq_dataset)

    # Infer
    output_root = cfg.TEST_OUTPUT_PATH
    output_mask_root = os.path.join(output_root, 'pred_masks')
    if not os.path.exists(output_mask_root):
        os.makedirs(output_mask_root)

    for seq_dataset in seq_datasets:
        seq_name = seq_dataset.seq_name
        image_seq_root = os.path.join(image_root, seq_name)
        output_mask_seq_root = os.path.join(output_mask_root, seq_name)
        if not os.path.exists(output_mask_seq_root):
            os.makedirs(output_mask_seq_root)
        print('Build a dataloader for sequence {}.'.format(seq_name))
        seq_dataloader = DataLoader(seq_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=cfg.TEST_WORKERS,
                                    pin_memory=True)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video_path = os.path.join(
            output_root, '{}_{}fps.avi'.format(seq_name, video_fps))

        print('Start the inference of sequence {}:'.format(seq_name))
        model.eval()
        engine.restart_engine()
        with torch.no_grad():
            for frame_idx, samples in enumerate(seq_dataloader):
                sample = samples[0]
                img_name = sample['meta']['current_name'][0]

                obj_nums = sample['meta']['obj_num']
                output_height = sample['meta']['height']
                output_width = sample['meta']['width']
                obj_idx = sample['meta']['obj_idx']

                obj_nums = [int(obj_num) for obj_num in obj_nums]
                obj_idx = [int(_obj_idx) for _obj_idx in obj_idx]

                current_img = sample['current_img']
                current_img = current_img.cuda(gpu_id, non_blocking=True)

                if frame_idx == 0:
                    videoWriter = cv2.VideoWriter(
                        output_video_path, fourcc, video_fps,
                        (int(output_width), int(output_height)))
                    print(
                        'Object number: {}. Inference size: {}x{}. Output size: {}x{}.'
                        .format(obj_nums[0],
                                current_img.size()[2],
                                current_img.size()[3], int(output_height),
                                int(output_width)))
                    current_label = sample['current_label'].cuda(
                        gpu_id, non_blocking=True).float()
                    current_label = F.interpolate(current_label,
                                                  size=current_img.size()[2:],
                                                  mode="nearest")
                    # add reference frame
                    engine.add_reference_frame(current_img,
                                               current_label,
                                               frame_step=0,
                                               obj_nums=obj_nums)
                else:
                    print('Processing image {}...'.format(img_name))
                    # predict segmentation
                    engine.match_propogate_one_frame(current_img)
                    pred_logit = engine.decode_current_logits(
                        (output_height, output_width))
                    pred_prob = torch.softmax(pred_logit, dim=1)
                    pred_label = torch.argmax(pred_prob, dim=1,
                                              keepdim=True).float()
                    _pred_label = F.interpolate(pred_label,
                                                size=engine.input_size_2d,
                                                mode="nearest")
                    # update memory
                    engine.update_memory(_pred_label)

                    # save results
                    input_image_path = os.path.join(image_seq_root, img_name)
                    output_mask_path = os.path.join(
                        output_mask_seq_root,
                        img_name.split('.')[0] + '.png')

                    pred_label = Image.fromarray(
                        pred_label.squeeze(0).squeeze(0).cpu().numpy().astype(
                            'uint8')).convert('P')
                    pred_label.putpalette(_palette)
                    pred_label.save(output_mask_path)

                    input_image = Image.open(input_image_path)

                    overlayed_image = overlay(
                        np.array(input_image, dtype=np.uint8),
                        np.array(pred_label, dtype=np.uint8), color_palette)
                    videoWriter.write(overlayed_image[..., [2, 1, 0]])

        print('Save a visualization video to {}.'.format(output_video_path))
        videoWriter.release()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AOT Demo")
    parser.add_argument('--exp_name', type=str, default='default')

    parser.add_argument('--stage', type=str, default='pre_ytb_dav')
    parser.add_argument('--model', type=str, default='r50_aotl')

    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--data_path', type=str, default='./datasets/Demo')
    parser.add_argument('--output_path', type=str, default='./demo_output')
    parser.add_argument('--ckpt_path',
                        type=str,
                        default='./pretrain_models/R50_AOTL_PRE_YTB_DAV.pth')

    parser.add_argument('--max_resolution', type=float, default=480 * 1.3)

    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=False)

    args = parser.parse_args()

    engine_config = importlib.import_module('configs.' + args.stage)
    cfg = engine_config.EngineConfig(args.exp_name, args.model)

    cfg.TEST_GPU_ID = args.gpu_id

    cfg.TEST_CKPT_PATH = args.ckpt_path
    cfg.TEST_DATA_PATH = args.data_path
    cfg.TEST_OUTPUT_PATH = args.output_path

    cfg.TEST_MIN_SIZE = None
    cfg.TEST_MAX_SIZE = args.max_resolution * 800. / 480.

    if args.amp:
        with torch.cuda.amp.autocast(enabled=True):
            demo(cfg)
    else:
        demo(cfg)


if __name__ == '__main__':
    main()
