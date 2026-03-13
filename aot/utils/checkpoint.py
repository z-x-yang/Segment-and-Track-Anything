import torch
import shutil
import numpy as np
from pathlib import Path


def get_device(gpu=None):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}" if gpu is not None else "cuda")
    return torch.device("cpu")


def load_network_and_optimizer(net, opt, pretrained_dir, gpu=None, scaler=None):
    device = get_device(gpu)
    pretrained = torch.load(pretrained_dir, map_location=device, weights_only=False)

    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()

    pretrained_dict_update = {}
    pretrained_dict_remove = []

    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k.startswith("module.") and k[7:] in model_dict:
            pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)

    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)

    opt.load_state_dict(pretrained['optimizer'])

    if scaler is not None and 'scaler' in pretrained:
        scaler.load_state_dict(pretrained['scaler'])

    del pretrained
    return net.to(device), opt, pretrained_dict_remove


def load_network_and_optimizer_v2(net, opt, pretrained_dir, gpu=None, scaler=None):
    device = get_device(gpu)
    pretrained = torch.load(pretrained_dir, map_location=device, weights_only=False)

    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()

    pretrained_dict_update = {}
    pretrained_dict_remove = []

    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k.startswith("module.") and k[7:] in model_dict:
            pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)

    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)

    opt_dict = opt.state_dict()

    all_params = {
        param_group['name']: param_group['params'][0]
        for param_group in opt_dict['param_groups']
    }

    pretrained_opt_dict = {'state': {}, 'param_groups': []}

    for param_group in pretrained['optimizer']['param_groups']:
        name = param_group['name']
        if name in all_params:
            param_id = all_params[name]

            pretrained_opt_dict['state'][param_id] = \
                pretrained['optimizer']['state'][param_group['params'][0]]

            param_group['params'][0] = param_id
            pretrained_opt_dict['param_groups'].append(param_group)

    opt_dict.update(pretrained_opt_dict)
    opt.load_state_dict(opt_dict)

    if scaler is not None and 'scaler' in pretrained:
        scaler.load_state_dict(pretrained['scaler'])

    del pretrained
    return net.to(device), opt, pretrained_dict_remove


def load_network(net, pretrained_dir, gpu=None):
    device = get_device(gpu)
    pretrained = torch.load(pretrained_dir, map_location=device, weights_only=False)

    if 'state_dict' in pretrained:
        pretrained_dict = pretrained['state_dict']
    elif 'model' in pretrained:
        pretrained_dict = pretrained['model']
    else:
        pretrained_dict = pretrained

    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []

    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k.startswith("module.") and k[7:] in model_dict:
            pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)

    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)

    del pretrained
    return net.to(device), pretrained_dict_remove


def save_network(net, opt, step, save_path, max_keep=8,
                 backup_dir='./saved_models', scaler=None):

    save_path = Path(save_path)
    backup_dir = Path(backup_dir)

    ckpt = {
        'state_dict': net.state_dict(),
        'optimizer': opt.state_dict()
    }

    if scaler is not None:
        ckpt['scaler'] = scaler.state_dict()

    try:
        save_path.mkdir(parents=True, exist_ok=True)
        save_file = save_path / f"save_step_{step}.pth"
        torch.save(ckpt, save_file)

    except Exception:
        backup_dir.mkdir(parents=True, exist_ok=True)
        save_file = backup_dir / f"save_step_{step}.pth"
        torch.save(ckpt, save_file)
        save_path = backup_dir

    all_ckpt = list(save_path.glob("save_step_*.pth"))

    if len(all_ckpt) > max_keep:

        steps = sorted([
            int(p.stem.split("_")[-1]) for p in all_ckpt
        ])[:-max_keep]

        for s in steps:
            ckpt_path = save_path / f"save_step_{s}.pth"
            if ckpt_path.exists():
                ckpt_path.unlink()


def cp_ckpt(remote_dir="data_wd/youtube_vos_jobs/result",
            curr_dir="backup"):

    remote_dir = Path(remote_dir)
    curr_dir = Path(curr_dir)

    for exp_dir in curr_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        for stage_dir in exp_dir.iterdir():
            finals = ["ema_ckpt", "ckpt"]

            for final in finals:
                final_dir = stage_dir / final
                if not final_dir.exists():
                    continue

                for ckpt in final_dir.glob("*.pth"):

                    remote_ckpt_path = remote_dir / exp_dir.name / stage_dir.name / final / ckpt.name

                    remote_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

                    if remote_ckpt_path.exists():
                        remote_ckpt_path.unlink()

                    try:
                        shutil.copy2(ckpt, remote_ckpt_path)
                        print(f"Copy {ckpt} to {remote_ckpt_path}")

                    except OSError as e:
                        print(e)
                        return