# -*- coding: utf-8 -*-
# @Time    : 6/23/21 3:19 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_sc.py

import numpy as np
import json
import os
import wget
from torchaudio.datasets import SPEECHCOMMANDS

# prepare the data of the speechcommands dataset.
print('Now download and process speechcommands dataset, it will take a few moments...')

# download the speechcommands dataset
if os.path.exists('./data/speech_commands_v0.02') == False:
    # we use the 35 class v2 dataset, which is used in torchaudio https://pytorch.org/audio/stable/_modules/torchaudio/datasets/speechcommands.html
    sc_url = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    wget.download(sc_url, out='./data/')
    os.mkdir('./data/speech_commands_v0.02')
    os.system('tar -xzvf ./data/speech_commands_v0.02.tar.gz -C ./data/speech_commands_v0.02')
    os.remove('./data/speech_commands_v0.02.tar.gz')

# generate training list = all samples - validation_list - testing_list
if os.path.exists('./data/speech_commands_v0.02/train_list.txt')==False:
    with open('./data/speech_commands_v0.02/validation_list.txt', 'r') as f:
        val_list = f.readlines()

    with open('./data/speech_commands_v0.02/testing_list.txt', 'r') as f:
        test_list = f.readlines()

    val_test_list = list(set(test_list+val_list))

    def get_immediate_subdirectories(a_dir):
        return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]
    def get_immediate_files(a_dir):
        return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

    base_path = './data/speech_commands_v0.02/'
    all_cmds = get_immediate_subdirectories(base_path)
    all_list = []
    for cmd in all_cmds:
        if cmd != '_background_noise_':
            cmd_samples = get_immediate_files(base_path+'/'+cmd)
            for sample in cmd_samples:
                all_list.append(cmd + '/' + sample+'\n')

    training_list = [x for x in all_list if x not in val_test_list]

    with open('./data/speech_commands_v0.02/train_list.txt', 'w') as f:
        f.writelines(training_list)

# The implementation of torchaudio has some bugs, use my own implementation, but the split results are exactly the same
# print('Now download and process speechcommands dataset, it will take a few moments...')
# class SubsetSC(SPEECHCOMMANDS):
#     def __init__(self, subset: str = None):
#         super().__init__("./data/", download=True)
#
#         def load_list(filename):
#             filepath = os.path.join(self._path, filename)
#             with open(filepath) as fileobj:
#                 return [os.path.join(self._path, line.strip()) for line in fileobj]
#
#         if subset == "validation":
#             self._walker = load_list("validation_list.txt")
#         elif subset == "testing":
#             self._walker = load_list("testing_list.txt")
#         elif subset == "training":
#             excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
#             excludes = set(excludes)
#             self._walker = [w for w in self._walker if w not in excludes]
#             train_full_path = [w for w in self._walker if w not in excludes]
#             gen_train_list(train_full_path)
#
# def gen_train_list(train_full_path):
#     train_list = []
#     for fullpath in train_full_path:
#         fullpath = fullpath.split('/')[3:]
#         fullpath = '/'.join(fullpath)+'\n'
#         train_list.append(fullpath)
#     with open('./data/SpeechCommands/speech_commands_v0.02/train_list.txt', 'w') as f:
#         f.writelines(train_list)

# Create training and testing split of the data. We do not use validation in this tutorial. Function borrowed from torchaudio implementation.
#train_set = SubsetSC("training")

label_set = np.loadtxt('./data/speechcommands_class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

# generate  json files
if os.path.exists('./data/datafiles') == False:
    os.mkdir('./data/datafiles')
    base_path = './data/speech_commands_v0.02/'
    for split in ['testing', 'validation', 'train']:
        wav_list = []
        with open(base_path+split+'_list.txt', 'r') as f:
            filelist = f.readlines()
        for file in filelist:
            cur_label = label_map[file.split('/')[0]]
            cur_path = os.path.abspath(os.getcwd()) + '/data/speech_commands_v0.02/' + file.strip()
            cur_dict = {"wav": cur_path, "labels": '/m/spcmd'+cur_label.zfill(2)}
            wav_list.append(cur_dict)
        if split == 'train':
            with open('./data/datafiles/speechcommand_train_data.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
        if split == 'testing':
            with open('./data/datafiles/speechcommand_eval_data.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
        if split == 'validation':
            with open('./data/datafiles/speechcommand_valid_data.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
        print(split + ' data processing finished, total {:d} samples'.format(len(wav_list)))

    print('Speechcommands dataset processing finished.')



