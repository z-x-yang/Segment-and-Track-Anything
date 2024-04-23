# -*- coding: utf-8 -*-
# @Time    : 6/23/21 5:38 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ensemble.py

# get the ensemble result

import os, sys, argparse
parentdir = str(os.path.abspath(os.path.join(__file__ ,"../../..")))+'/src'
sys.path.append(parentdir)

import dataloader
import models
from utilities import *
from traintest import train, validate
import numpy as np
from scipy import stats
import torch

eval_data_path = '/data/sls/scratch/yuangong/audioset/datafiles/eval_data.json'

def get_ensemble_res(mdl_list, base_path):
    # the 0-len(mdl_list) rows record the results of single models, the last row record the result of the ensemble model.
    ensemble_res = np.zeros([len(mdl_list)+1, 3])
    if os.path.exists(base_path) == False:
        os.mkdir(base_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_idx, mdl in enumerate(mdl_list):
        print('-----------------------')
        print('now loading model {:d}: {:s}'.format(model_idx, mdl))

        # sd = torch.load('/Users/yuan/Documents/ast/pretrained_models/audio_model_wa.pth', map_location=device)
        sd = torch.load(mdl, map_location=device)
        # get the time and freq stride of the pretrained model
        fstride, tstride = int(mdl.split('/')[-1].split('_')[1]), int(mdl.split('/')[-1].split('_')[2].split('.')[0])
        audio_model = models.ASTModel(fstride=fstride, tstride=tstride)
        audio_model = torch.nn.DataParallel(audio_model)
        audio_model.load_state_dict(sd, strict=False)

        args.exp_dir = base_path

        stats, _ = validate(audio_model, eval_loader, args, model_idx)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        dprime = d_prime(mAUC)
        ensemble_res[model_idx, :] = [mAP, mAUC, dprime]
        print("Model {:d} {:s} mAP: {:.6f}, AUC: {:.6f}, d-prime: {:.6f}".format(model_idx, mdl, mAP, mAUC, dprime))

    # calculate the ensemble result
    # get the ground truth label
    target = np.loadtxt(base_path + '/predictions/target.csv', delimiter=',')
    # get the ground truth label
    prediction_sample = np.loadtxt(base_path + '/predictions/predictions_0.csv', delimiter=',')
    # allocate memory space for the ensemble prediction
    predictions_table = np.zeros([len(mdl_list) , prediction_sample.shape[0], prediction_sample.shape[1]])
    for model_idx in range(0, len(mdl_list)):
        predictions_table[model_idx, :, :] = np.loadtxt(base_path + '/predictions/predictions_' + str(model_idx) + '.csv', delimiter=',')
        model_idx += 1

    ensemble_predictions = np.mean(predictions_table, axis=0)
    stats = calculate_stats(ensemble_predictions, target)
    ensemble_mAP = np.mean([stat['AP'] for stat in stats])
    ensemble_mAUC = np.mean([stat['auc'] for stat in stats])
    ensemble_dprime = d_prime(ensemble_mAUC)
    ensemble_res[-1, :] = [ensemble_mAP, ensemble_mAUC, ensemble_dprime]
    print('---------------Ensemble Result Summary---------------')
    for model_idx in range(len(mdl_list)):
        print("Model {:d} {:s} mAP: {:.6f}, AUC: {:.6f}, d-prime: {:.6f}".format(model_idx, mdl_list[model_idx], ensemble_res[model_idx, 0], ensemble_res[model_idx, 1], ensemble_res[model_idx, 2]))
    print("Ensemble {:d} Models mAP: {:.6f}, AUC: {:.6f}, d-prime: {:.6f}".format(len(mdl_list), ensemble_mAP, ensemble_mAUC, ensemble_dprime))
    np.savetxt(base_path + '/ensemble_result.csv', ensemble_res, delimiter=',')

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

# dataloader settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = parser.parse_args()
args.dataset='audioset'
args.data_eval= eval_data_path
args.label_csv='/data/sls/scratch/yuangong/ast/egs/audioset/class_labels_indices.csv'
args.loss_fn = torch.nn.BCEWithLogitsLoss()
norm_stats = {'audioset': [-4.2677393, 4.5689974], 'esc50': [-6.6268077, 5.358466],
              'speechcommands': [-6.845978, 5.5654526]}
target_length = {'audioset': 1024, 'esc50': 512, 'speechcommands': 128}
noise = {'audioset': False, 'esc50': False, 'speechcommands': True}

val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1], 'noise':False}
eval_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=100, shuffle=False, num_workers=16, pin_memory=True)


# formal full ensemble, ensemble-S
mdl_list_s = ['/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_10_10_0.4495.pth',
'/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_10_10_0.4483.pth',
'/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_10_10_0.4475.pth']

# formal full ensemble, ensemble-M
mdl_list_m = ['/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_10_10_0.4495.pth',
'/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_10_10_0.4483.pth',
'/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_10_10_0.4475.pth',
'/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_12_12_0.4467.pth',
'/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_14_14_0.4431.pth',
'/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_16_16_0.4422.pth']

# ensemble 3 models that is trained with same setting, but different random seeds
get_ensemble_res(mdl_list_s, './exp/ensemble_s')
# ensemble 6 models that is trained with different settings (3 with stride of 10, others are with stride of 12, 14, and 16)
get_ensemble_res(mdl_list_m, './exp/ensemble_m')