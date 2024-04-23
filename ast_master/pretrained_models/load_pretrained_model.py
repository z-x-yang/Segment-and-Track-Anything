# -*- coding: utf-8 -*-
# @Time    : 6/28/21 4:54 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : load_pretrained_model.py

# sample code of loading a pretrained AST model

import os, sys
parentdir = str(os.path.abspath(os.path.join(__file__ ,"../../")))+'/src'
print(parentdir)
sys.path.append(parentdir)
import models
import torch

# audioset input sequence length is 1024
pretrained_mdl_path = '../../pretrained_models/audioset_10_10_0.4593.pth'
# get the frequency and time stride of the pretrained model from its name
fstride, tstride = int(pretrained_mdl_path.split('/')[-1].split('_')[1]), int(pretrained_mdl_path.split('/')[-1].split('_')[2].split('.')[0])
# The input of audioset pretrained model is 1024 frames.
input_tdim = 1024

# initialize an AST model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load(pretrained_mdl_path, map_location=device)
audio_model = models.ASTModel(input_tdim=input_tdim, fstride=fstride, tstride=tstride)
audio_model = torch.nn.DataParallel(audio_model)
audio_model.load_state_dict(sd, strict=False)

input_tdim = 1024
# input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
test_input = torch.rand([10, input_tdim, 128])
test_output = audio_model(test_input)
# output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
print(test_output.shape)