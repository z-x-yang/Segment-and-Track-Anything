import torch
import os
import shutil
import numpy as np


def load_network_and_optimizer(net, opt, pretrained_dir, gpu, scaler=None):
    pretrained = torch.load(pretrained_dir,
                            map_location=torch.device("cuda:" + str(gpu)))
    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    opt.load_state_dict(pretrained['optimizer'])
    if scaler is not None and 'scaler' in pretrained.keys():
        scaler.load_state_dict(pretrained['scaler'])
    del (pretrained)
    return net.cuda(gpu), opt, pretrained_dict_remove


def load_network_and_optimizer_v2(net, opt, pretrained_dir, gpu, scaler=None):
    pretrained = torch.load(pretrained_dir,
                            map_location=torch.device("cuda:" + str(gpu)))
    # load model
    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)

    # load optimizer
    opt_dict = opt.state_dict()
    all_params = {
        param_group['name']: param_group['params'][0]
        for param_group in opt_dict['param_groups']
    }
    pretrained_opt_dict = {'state': {}, 'param_groups': []}
    for idx in range(len(pretrained['optimizer']['param_groups'])):
        param_group = pretrained['optimizer']['param_groups'][idx]
        if param_group['name'] in all_params.keys():
            pretrained_opt_dict['state'][all_params[
                param_group['name']]] = pretrained['optimizer']['state'][
                    param_group['params'][0]]
            param_group['params'][0] = all_params[param_group['name']]
            pretrained_opt_dict['param_groups'].append(param_group)

    opt_dict.update(pretrained_opt_dict)
    opt.load_state_dict(opt_dict)

    # load scaler
    if scaler is not None and 'scaler' in pretrained.keys():
        scaler.load_state_dict(pretrained['scaler'])
    del (pretrained)
    return net.cuda(gpu), opt, pretrained_dict_remove


def load_network(net, pretrained_dir, gpu):
    pretrained = torch.load(pretrained_dir,
                            map_location=torch.device("cuda:" + str(gpu)))
    if 'state_dict' in pretrained.keys():
        pretrained_dict = pretrained['state_dict']
    elif 'model' in pretrained.keys():
        pretrained_dict = pretrained['model']
    else:
        pretrained_dict = pretrained
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    del (pretrained)
    return net.cuda(gpu), pretrained_dict_remove


def save_network(net,
                 opt,
                 step,
                 save_path,
                 max_keep=8,
                 backup_dir='./saved_models',
                 scaler=None):
    ckpt = {'state_dict': net.state_dict(), 'optimizer': opt.state_dict()}
    if scaler is not None:
        ckpt['scaler'] = scaler.state_dict()

    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = 'save_step_%s.pth' % (step)
        save_dir = os.path.join(save_path, save_file)
        torch.save(ckpt, save_dir)
    except:
        save_path = backup_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = 'save_step_%s.pth' % (step)
        save_dir = os.path.join(save_path, save_file)
        torch.save(ckpt, save_dir)

    all_ckpt = os.listdir(save_path)
    if len(all_ckpt) > max_keep:
        all_step = []
        for ckpt_name in all_ckpt:
            step = int(ckpt_name.split('_')[-1].split('.')[0])
            all_step.append(step)
        all_step = list(np.sort(all_step))[:-max_keep]
        for step in all_step:
            ckpt_path = os.path.join(save_path, 'save_step_%s.pth' % (step))
            os.system('rm {}'.format(ckpt_path))


def cp_ckpt(remote_dir="data_wd/youtube_vos_jobs/result", curr_dir="backup"):
    exps = os.listdir(curr_dir)
    for exp in exps:
        exp_dir = os.path.join(curr_dir, exp)
        stages = os.listdir(exp_dir)
        for stage in stages:
            stage_dir = os.path.join(exp_dir, stage)
            finals = ["ema_ckpt", "ckpt"]
            for final in finals:
                final_dir = os.path.join(stage_dir, final)
                ckpts = os.listdir(final_dir)
                for ckpt in ckpts:
                    if '.pth' not in ckpt:
                        continue
                    curr_ckpt_path = os.path.join(final_dir, ckpt)
                    remote_ckpt_path = os.path.join(remote_dir, exp, stage,
                                                    final, ckpt)
                    if os.path.exists(remote_ckpt_path):
                        os.system('rm {}'.format(remote_ckpt_path))
                    try:
                        shutil.copy(curr_ckpt_path, remote_ckpt_path)
                        print("Copy {} to {}.".format(curr_ckpt_path,
                                                      remote_ckpt_path))
                    except OSError as Inst:
                        return
