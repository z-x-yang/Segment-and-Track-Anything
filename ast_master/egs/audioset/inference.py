# -*- coding: utf-8 -*-
# @Author   : jeffcheng
# @Time     : 2021/9/1 - 15:13
# @Reference: a inference script for single audio, heavily base on demo.py and traintest.py
import os
import sys
import csv
import argparse

import numpy as np
import torch
import torchaudio

torchaudio.set_audio_backend("soundfile")       # switch backend
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from src.models import ASTModel

# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'


def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser:'
                                                 'python inference --audio_path ./0OxlgIitVig.wav '
                                                 '--model_path ./pretrained_models/audioset_10_10_0.4593.pth')

    parser.add_argument("--model_path", type=str, required=True,
                        help="the trained model you want to test")
    parser.add_argument('--audio_path',
                        help='the audio you want to predict, sample rate 16k.',
                        type=str, required=True)

    args = parser.parse_args()

    label_csv = './data/class_labels_indices.csv'       # label and indices for audioset data

    # 1. make feature for predict
    audio_path = args.audio_path
    feats = make_features(audio_path, mel_bins=128)           # shape(1024, 128)

    # assume each input spectrogram has 100 time frames
    input_tdim = feats.shape[0]

    # 2. load the best model and the weights
    checkpoint_path = args.model_path
    ast_mdl = ASTModel(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
    print(f'[*INFO] load checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)

    audio_model = audio_model.to(torch.device("cuda:0"))

    # 3. feed the data feature to model
    feats_data = feats.expand(1, input_tdim, 128)           # reshape the feature

    audio_model.eval()                                      # set the eval model
    with torch.no_grad():
        output = audio_model.forward(feats_data)
        output = torch.sigmoid(output)
    result_output = output.data.cpu().numpy()[0]

    # 4. map the post-prob to label
    labels = load_label(label_csv)

    sorted_indexes = np.argsort(result_output)[::-1]

    # Print audio tagging top probabilities
    print('[*INFO] predice results:')
    for k in range(10):
        print('{}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]],
                                  result_output[sorted_indexes[k]]))
