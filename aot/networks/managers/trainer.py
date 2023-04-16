import os
import time
import json
import datetime as datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.train_datasets import DAVIS2017_Train, YOUTUBEVOS_Train, StaticTrain, TEST
import dataloaders.video_transforms as tr

from utils.meters import AverageMeter
from utils.image import label2colormap, masked_image, save_image
from utils.checkpoint import load_network_and_optimizer, load_network, save_network
from utils.learning import adjust_learning_rate, get_trainable_params
from utils.metric import pytorch_iou
from utils.ema import ExponentialMovingAverage, get_param_buffer_for_ema

from networks.models import build_vos_model
from networks.engines import build_engine


class Trainer(object):
    def __init__(self, rank, cfg, enable_amp=True):
        self.gpu = rank + cfg.DIST_START_GPU
        self.gpu_num = cfg.TRAIN_GPUS 
        self.rank = rank
        self.cfg = cfg

        self.print_log("Exp {}:".format(cfg.EXP_NAME))
        self.print_log(json.dumps(cfg.__dict__, indent=4, sort_keys=True))

        print("Use GPU {} for training VOS.".format(self.gpu))
        torch.cuda.set_device(self.gpu)
        torch.backends.cudnn.benchmark = True if cfg.DATA_RANDOMCROP[
            0] == cfg.DATA_RANDOMCROP[
                1] and 'swin' not in cfg.MODEL_ENCODER else False

        self.print_log('Build VOS model.')

        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(self.gpu)
        self.model_encoder = self.model.encoder
        self.engine = build_engine(
            cfg.MODEL_ENGINE,
            'train',
            aot_model=self.model,
            gpu_id=self.gpu,
            long_term_mem_gap=cfg.TRAIN_LONG_TERM_MEM_GAP)

        if cfg.MODEL_FREEZE_BACKBONE:
            for param in self.model_encoder.parameters():
                param.requires_grad = False

        if cfg.DIST_ENABLE:
            dist.init_process_group(backend=cfg.DIST_BACKEND,
                                    init_method=cfg.DIST_URL,
                                    world_size=cfg.TRAIN_GPUS,
                                    rank=rank,
                                    timeout=datetime.timedelta(seconds=300))

            self.model.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model.encoder).cuda(self.gpu)

            self.dist_engine = torch.nn.parallel.DistributedDataParallel(
                self.engine,
                device_ids=[self.gpu],
                output_device=self.gpu,
                find_unused_parameters=True,
                broadcast_buffers=False)
        else:
            self.dist_engine = self.engine

        self.use_frozen_bn = False
        if 'swin' in cfg.MODEL_ENCODER:
            self.print_log('Use LN in Encoder!')
        elif not cfg.MODEL_FREEZE_BN:
            if cfg.DIST_ENABLE:
                self.print_log('Use Sync BN in Encoder!')
            else:
                self.print_log('Use BN in Encoder!')
        else:
            self.use_frozen_bn = True
            self.print_log('Use Frozen BN in Encoder!')

        if self.rank == 0:
            try:
                total_steps = float(cfg.TRAIN_TOTAL_STEPS)
                ema_decay = 1. - 1. / (total_steps * cfg.TRAIN_EMA_RATIO)
                self.ema_params = get_param_buffer_for_ema(
                    self.model, update_buffer=(not cfg.MODEL_FREEZE_BN))
                self.ema = ExponentialMovingAverage(self.ema_params,
                                                    decay=ema_decay)
                self.ema_dir = cfg.DIR_EMA_CKPT
            except Exception as inst:
                self.print_log(inst)
                self.print_log('Error: failed to create EMA model!')

        self.print_log('Build optimizer.')

        trainable_params = get_trainable_params(
            model=self.dist_engine,
            base_lr=cfg.TRAIN_LR,
            use_frozen_bn=self.use_frozen_bn,
            weight_decay=cfg.TRAIN_WEIGHT_DECAY,
            exclusive_wd_dict=cfg.TRAIN_WEIGHT_DECAY_EXCLUSIVE,
            no_wd_keys=cfg.TRAIN_WEIGHT_DECAY_EXEMPTION)

        if cfg.TRAIN_OPT == 'sgd':
            self.optimizer = optim.SGD(trainable_params,
                                       lr=cfg.TRAIN_LR,
                                       momentum=cfg.TRAIN_SGD_MOMENTUM,
                                       nesterov=True)
        else:
            self.optimizer = optim.AdamW(trainable_params,
                                         lr=cfg.TRAIN_LR,
                                         weight_decay=cfg.TRAIN_WEIGHT_DECAY)

        self.enable_amp = enable_amp
        if enable_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.prepare_dataset()
        self.process_pretrained_model()

        if cfg.TRAIN_TBLOG and self.rank == 0:
            from tensorboardX import SummaryWriter
            self.tblogger = SummaryWriter(cfg.DIR_TB_LOG)

    def process_pretrained_model(self):
        cfg = self.cfg

        self.step = cfg.TRAIN_START_STEP
        self.epoch = 0

        if cfg.TRAIN_AUTO_RESUME:
            ckpts = os.listdir(cfg.DIR_CKPT)
            if len(ckpts) > 0:
                ckpts = list(
                    map(lambda x: int(x.split('_')[-1].split('.')[0]), ckpts))
                ckpt = np.sort(ckpts)[-1]
                cfg.TRAIN_RESUME = True
                cfg.TRAIN_RESUME_CKPT = ckpt
                cfg.TRAIN_RESUME_STEP = ckpt
            else:
                cfg.TRAIN_RESUME = False

        if cfg.TRAIN_RESUME:
            if self.rank == 0:
                try:
                    try:
                        ema_ckpt_dir = os.path.join(
                            self.ema_dir,
                            'save_step_%s.pth' % (cfg.TRAIN_RESUME_CKPT))
                        ema_model, removed_dict = load_network(
                            self.model, ema_ckpt_dir, self.gpu)
                    except Exception as inst:
                        self.print_log(inst)
                        self.print_log('Try to use backup EMA checkpoint.')
                        DIR_RESULT = './backup/{}/{}'.format(
                            cfg.EXP_NAME, cfg.STAGE_NAME)
                        DIR_EMA_CKPT = os.path.join(DIR_RESULT, 'ema_ckpt')
                        ema_ckpt_dir = os.path.join(
                            DIR_EMA_CKPT,
                            'save_step_%s.pth' % (cfg.TRAIN_RESUME_CKPT))
                        ema_model, removed_dict = load_network(
                            self.model, ema_ckpt_dir, self.gpu)

                    if len(removed_dict) > 0:
                        self.print_log(
                            'Remove {} from EMA model.'.format(removed_dict))
                    ema_decay = self.ema.decay
                    del (self.ema)

                    ema_params = get_param_buffer_for_ema(
                        ema_model, update_buffer=(not cfg.MODEL_FREEZE_BN))
                    self.ema = ExponentialMovingAverage(ema_params,
                                                        decay=ema_decay)
                    self.ema.num_updates = cfg.TRAIN_RESUME_CKPT
                except Exception as inst:
                    self.print_log(inst)
                    self.print_log('Error: EMA model not found!')

            try:
                resume_ckpt = os.path.join(
                    cfg.DIR_CKPT, 'save_step_%s.pth' % (cfg.TRAIN_RESUME_CKPT))
                self.model, self.optimizer, removed_dict = load_network_and_optimizer(
                    self.model,
                    self.optimizer,
                    resume_ckpt,
                    self.gpu,
                    scaler=self.scaler)
            except Exception as inst:
                self.print_log(inst)
                self.print_log('Try to use backup checkpoint.')
                DIR_RESULT = './backup/{}/{}'.format(cfg.EXP_NAME,
                                                     cfg.STAGE_NAME)
                DIR_CKPT = os.path.join(DIR_RESULT, 'ckpt')
                resume_ckpt = os.path.join(
                    DIR_CKPT, 'save_step_%s.pth' % (cfg.TRAIN_RESUME_CKPT))
                self.model, self.optimizer, removed_dict = load_network_and_optimizer(
                    self.model,
                    self.optimizer,
                    resume_ckpt,
                    self.gpu,
                    scaler=self.scaler)

            if len(removed_dict) > 0:
                self.print_log(
                    'Remove {} from checkpoint.'.format(removed_dict))

            self.step = cfg.TRAIN_RESUME_STEP
            if cfg.TRAIN_TOTAL_STEPS <= self.step:
                self.print_log("Your training has finished!")
                exit()
            self.epoch = int(np.ceil(self.step / len(self.train_loader)))

            self.print_log('Resume from step {}'.format(self.step))

        elif cfg.PRETRAIN:
            if cfg.PRETRAIN_FULL:
                try:
                    self.model, removed_dict = load_network(
                        self.model, cfg.PRETRAIN_MODEL, self.gpu)
                except Exception as inst:
                    self.print_log(inst)
                    self.print_log('Try to use backup EMA checkpoint.')
                    DIR_RESULT = './backup/{}/{}'.format(
                        cfg.EXP_NAME, cfg.STAGE_NAME)
                    DIR_EMA_CKPT = os.path.join(DIR_RESULT, 'ema_ckpt')
                    PRETRAIN_MODEL = os.path.join(
                        DIR_EMA_CKPT,
                        cfg.PRETRAIN_MODEL.split('/')[-1])
                    self.model, removed_dict = load_network(
                        self.model, PRETRAIN_MODEL, self.gpu)

                if len(removed_dict) > 0:
                    self.print_log('Remove {} from pretrained model.'.format(
                        removed_dict))
                self.print_log('Load pretrained VOS model from {}.'.format(
                    cfg.PRETRAIN_MODEL))
            else:
                model_encoder, removed_dict = load_network(
                    self.model_encoder, cfg.PRETRAIN_MODEL, self.gpu)
                if len(removed_dict) > 0:
                    self.print_log('Remove {} from pretrained model.'.format(
                        removed_dict))
                self.print_log(
                    'Load pretrained backbone model from {}.'.format(
                        cfg.PRETRAIN_MODEL))

    def prepare_dataset(self):
        cfg = self.cfg
        self.enable_prev_frame = cfg.TRAIN_ENABLE_PREV_FRAME

        self.print_log('Process dataset...')
        if cfg.TRAIN_AUG_TYPE == 'v1':
            composed_transforms = transforms.Compose([
                tr.RandomScale(cfg.DATA_MIN_SCALE_FACTOR,
                               cfg.DATA_MAX_SCALE_FACTOR,
                               cfg.DATA_SHORT_EDGE_LEN),
                tr.BalancedRandomCrop(cfg.DATA_RANDOMCROP,
                                      max_obj_num=cfg.MODEL_MAX_OBJ_NUM),
                tr.RandomHorizontalFlip(cfg.DATA_RANDOMFLIP),
                tr.Resize(cfg.DATA_RANDOMCROP, use_padding=True),
                tr.ToTensor()
            ])
        elif cfg.TRAIN_AUG_TYPE == 'v2':
            composed_transforms = transforms.Compose([
                tr.RandomScale(cfg.DATA_MIN_SCALE_FACTOR,
                               cfg.DATA_MAX_SCALE_FACTOR,
                               cfg.DATA_SHORT_EDGE_LEN),
                tr.BalancedRandomCrop(cfg.DATA_RANDOMCROP,
                                      max_obj_num=cfg.MODEL_MAX_OBJ_NUM),
                tr.RandomColorJitter(),
                tr.RandomGrayScale(),
                tr.RandomGaussianBlur(),
                tr.RandomHorizontalFlip(cfg.DATA_RANDOMFLIP),
                tr.Resize(cfg.DATA_RANDOMCROP, use_padding=True),
                tr.ToTensor()
            ])
        else:
            assert NotImplementedError

        train_datasets = []
        if 'static' in cfg.DATASETS:
            pretrain_vos_dataset = StaticTrain(
                cfg.DIR_STATIC,
                cfg.DATA_RANDOMCROP,
                seq_len=cfg.DATA_SEQ_LEN,
                merge_prob=cfg.DATA_DYNAMIC_MERGE_PROB,
                max_obj_n=cfg.MODEL_MAX_OBJ_NUM,
                aug_type=cfg.TRAIN_AUG_TYPE)
            train_datasets.append(pretrain_vos_dataset)
            self.enable_prev_frame = False

        if 'davis2017' in cfg.DATASETS:
            train_davis_dataset = DAVIS2017_Train(
                root=cfg.DIR_DAVIS,
                full_resolution=cfg.TRAIN_DATASET_FULL_RESOLUTION,
                transform=composed_transforms,
                repeat_time=cfg.DATA_DAVIS_REPEAT,
                seq_len=cfg.DATA_SEQ_LEN,
                rand_gap=cfg.DATA_RANDOM_GAP_DAVIS,
                rand_reverse=cfg.DATA_RANDOM_REVERSE_SEQ,
                merge_prob=cfg.DATA_DYNAMIC_MERGE_PROB,
                enable_prev_frame=self.enable_prev_frame,
                max_obj_n=cfg.MODEL_MAX_OBJ_NUM)
            train_datasets.append(train_davis_dataset)

        if 'youtubevos' in cfg.DATASETS:
            train_ytb_dataset = YOUTUBEVOS_Train(
                root=cfg.DIR_YTB,
                transform=composed_transforms,
                seq_len=cfg.DATA_SEQ_LEN,
                rand_gap=cfg.DATA_RANDOM_GAP_YTB,
                rand_reverse=cfg.DATA_RANDOM_REVERSE_SEQ,
                merge_prob=cfg.DATA_DYNAMIC_MERGE_PROB,
                enable_prev_frame=self.enable_prev_frame,
                max_obj_n=cfg.MODEL_MAX_OBJ_NUM)
            train_datasets.append(train_ytb_dataset)

        if 'test' in cfg.DATASETS:
            test_dataset = TEST(transform=composed_transforms,
                                seq_len=cfg.DATA_SEQ_LEN)
            train_datasets.append(test_dataset)

        if len(train_datasets) > 1:
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        elif len(train_datasets) == 1:
            train_dataset = train_datasets[0]
        else:
            self.print_log('No dataset!')
            exit(0)

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset) if self.cfg.DIST_ENABLE else None
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=int(cfg.TRAIN_BATCH_SIZE /
                                                      cfg.TRAIN_GPUS),
                                       shuffle=False if self.cfg.DIST_ENABLE else True,
                                       num_workers=cfg.DATA_WORKERS,
                                       pin_memory=True,
                                       sampler=self.train_sampler,
                                       drop_last=True,
                                       prefetch_factor=4)

        self.print_log('Done!')

    def sequential_training(self):

        cfg = self.cfg

        if self.enable_prev_frame:
            frame_names = ['Ref', 'Prev']
        else:
            frame_names = ['Ref(Prev)']

        for i in range(cfg.DATA_SEQ_LEN - 1):
            frame_names.append('Curr{}'.format(i + 1))

        seq_len = len(frame_names)

        running_losses = []
        running_ious = []
        for _ in range(seq_len):
            running_losses.append(AverageMeter())
            running_ious.append(AverageMeter())
        batch_time = AverageMeter()
        avg_obj = AverageMeter()

        optimizer = self.optimizer
        model = self.dist_engine
        train_sampler = self.train_sampler
        train_loader = self.train_loader
        step = self.step
        epoch = self.epoch
        max_itr = cfg.TRAIN_TOTAL_STEPS
        start_seq_training_step = int(cfg.TRAIN_SEQ_TRAINING_START_RATIO *
                                      max_itr)
        use_prev_prob = cfg.MODEL_USE_PREV_PROB

        self.print_log('Start training:')
        model.train()
        while step < cfg.TRAIN_TOTAL_STEPS:
            if self.cfg.DIST_ENABLE:
                train_sampler.set_epoch(epoch)
            epoch += 1
            last_time = time.time()
            for frame_idx, sample in enumerate(train_loader):
                if step > cfg.TRAIN_TOTAL_STEPS:
                    break

                if step % cfg.TRAIN_TBLOG_STEP == 0 and self.rank == 0 and cfg.TRAIN_TBLOG:
                    tf_board = True
                else:
                    tf_board = False

                if step >= start_seq_training_step:
                    use_prev_pred = True
                    freeze_params = cfg.TRAIN_SEQ_TRAINING_FREEZE_PARAMS
                else:
                    use_prev_pred = False
                    freeze_params = []

                if step % cfg.TRAIN_LR_UPDATE_STEP == 0:
                    now_lr = adjust_learning_rate(
                        optimizer=optimizer,
                        base_lr=cfg.TRAIN_LR,
                        p=cfg.TRAIN_LR_POWER,
                        itr=step,
                        max_itr=max_itr,
                        restart=cfg.TRAIN_LR_RESTART,
                        warm_up_steps=cfg.TRAIN_LR_WARM_UP_RATIO * max_itr,
                        is_cosine_decay=cfg.TRAIN_LR_COSINE_DECAY,
                        min_lr=cfg.TRAIN_LR_MIN,
                        encoder_lr_ratio=cfg.TRAIN_LR_ENCODER_RATIO,
                        freeze_params=freeze_params)

                ref_imgs = sample['ref_img']  # batch_size * 3 * h * w
                prev_imgs = sample['prev_img']
                curr_imgs = sample['curr_img']
                ref_labels = sample['ref_label']  # batch_size * 1 * h * w
                prev_labels = sample['prev_label']
                curr_labels = sample['curr_label']
                obj_nums = sample['meta']['obj_num']
                bs, _, h, w = curr_imgs[0].size()

                ref_imgs = ref_imgs.cuda(self.gpu, non_blocking=True)
                prev_imgs = prev_imgs.cuda(self.gpu, non_blocking=True)
                curr_imgs = [
                    curr_img.cuda(self.gpu, non_blocking=True)
                    for curr_img in curr_imgs
                ]
                ref_labels = ref_labels.cuda(self.gpu, non_blocking=True)
                prev_labels = prev_labels.cuda(self.gpu, non_blocking=True)
                curr_labels = [
                    curr_label.cuda(self.gpu, non_blocking=True)
                    for curr_label in curr_labels
                ]
                obj_nums = list(obj_nums)
                obj_nums = [int(obj_num) for obj_num in obj_nums]

                batch_size = ref_imgs.size(0)

                all_frames = torch.cat([ref_imgs, prev_imgs] + curr_imgs,
                                       dim=0)
                all_labels = torch.cat([ref_labels, prev_labels] + curr_labels,
                                       dim=0)

                self.engine.restart_engine(batch_size, True)
                optimizer.zero_grad(set_to_none=True)

                if self.enable_amp:
                    with torch.cuda.amp.autocast(enabled=True):
                        
                        loss, all_pred, all_loss, boards = model(
                            all_frames,
                            all_labels,
                            batch_size,
                            use_prev_pred=use_prev_pred,
                            obj_nums=obj_nums,
                            step=step,
                            tf_board=tf_board,
                            enable_prev_frame=self.enable_prev_frame,
                            use_prev_prob=use_prev_prob)
                        loss = torch.mean(loss)
                        
                    start = time.time()
                    self.scaler.scale(loss).backward()
                    end = time.time()
                    print(end-start)
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg.TRAIN_CLIP_GRAD_NORM)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    
                else:
                    loss, all_pred, all_loss, boards = model(
                        all_frames,
                        all_labels,
                        ref_imgs.size(0),
                        use_prev_pred=use_prev_pred,
                        obj_nums=obj_nums,
                        step=step,
                        tf_board=tf_board,
                        enable_prev_frame=self.enable_prev_frame,
                        use_prev_prob=use_prev_prob)
                    loss = torch.mean(loss)

                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg.TRAIN_CLIP_GRAD_NORM)
                    loss.backward()
                    optimizer.step()

                for idx in range(seq_len):
                    now_pred = all_pred[idx].detach()
                    now_label = all_labels[idx * bs:(idx + 1) * bs].detach()
                    now_loss = torch.mean(all_loss[idx].detach())
                    now_iou = pytorch_iou(now_pred.unsqueeze(1), now_label,
                                          obj_nums) * 100
                    if self.cfg.DIST_ENABLE:
                        dist.all_reduce(now_loss)
                        dist.all_reduce(now_iou)
                        now_loss /= self.gpu_num
                        now_iou /= self.gpu_num
                    if self.rank == 0:
                        running_losses[idx].update(now_loss.item())
                        running_ious[idx].update(now_iou.item())

                if self.rank == 0:
                    self.ema.update(self.ema_params)

                    avg_obj.update(sum(obj_nums) / float(len(obj_nums)))
                    curr_time = time.time()
                    batch_time.update(curr_time - last_time)
                    last_time = curr_time

                    if step % cfg.TRAIN_TBLOG_STEP == 0:
                        all_f = [ref_imgs, prev_imgs] + curr_imgs
                        self.process_log(ref_imgs, all_f[-2], all_f[-1],
                                         ref_labels, all_pred[-2], now_label,
                                         now_pred, boards, running_losses,
                                         running_ious, now_lr, step)

                    if step % cfg.TRAIN_LOG_STEP == 0:
                        strs = 'I:{}, LR:{:.5f}, T:{:.1f}({:.1f})s, Obj:{:.1f}({:.1f})'.format(
                            step, now_lr, batch_time.val,
                            batch_time.moving_avg, avg_obj.val,
                            avg_obj.moving_avg)
                        batch_time.reset()
                        avg_obj.reset()
                        for idx in range(seq_len):
                            strs += ', {}: L {:.3f}({:.3f}) IoU {:.1f}({:.1f})%'.format(
                                frame_names[idx], running_losses[idx].val,
                                running_losses[idx].moving_avg,
                                running_ious[idx].val,
                                running_ious[idx].moving_avg)
                            running_losses[idx].reset()
                            running_ious[idx].reset()

                        self.print_log(strs)

                step += 1

                if step % cfg.TRAIN_SAVE_STEP == 0 and self.rank == 0:
                    max_mem = torch.cuda.max_memory_allocated(
                        device=self.gpu) / (1024.**3)
                    ETA = str(
                        datetime.timedelta(
                            seconds=int(batch_time.moving_avg *
                                        (cfg.TRAIN_TOTAL_STEPS - step))))
                    self.print_log('ETA: {}, Max Mem: {:.2f}G.'.format(
                        ETA, max_mem))
                    self.print_log('Save CKPT (Step {}).'.format(step))
                    save_network(self.model,
                                 optimizer,
                                 step,
                                 cfg.DIR_CKPT,
                                 cfg.TRAIN_MAX_KEEP_CKPT,
                                 backup_dir='./backup/{}/{}/ckpt'.format(
                                     cfg.EXP_NAME, cfg.STAGE_NAME),
                                 scaler=self.scaler)
                    try:
                        torch.cuda.empty_cache()
                        # First save original parameters before replacing with EMA version
                        self.ema.store(self.ema_params)
                        # Copy EMA parameters to model
                        self.ema.copy_to(self.ema_params)
                        # Save EMA model
                        save_network(
                            self.model,
                            optimizer,
                            step,
                            self.ema_dir,
                            cfg.TRAIN_MAX_KEEP_CKPT,
                            backup_dir='./backup/{}/{}/ema_ckpt'.format(
                                cfg.EXP_NAME, cfg.STAGE_NAME),
                            scaler=self.scaler)
                        # Restore original parameters to resume training later
                        self.ema.restore(self.ema_params)
                    except Exception as inst:
                        self.print_log(inst)
                        self.print_log('Error: failed to save EMA model!')

        self.print_log('Stop training!')

    def print_log(self, string):
        if self.rank == 0:
            print(string)

    def process_log(self, ref_imgs, prev_imgs, curr_imgs, ref_labels,
                    prev_labels, curr_labels, curr_pred, boards,
                    running_losses, running_ious, now_lr, step):
        cfg = self.cfg

        mean = np.array([[[0.485]], [[0.456]], [[0.406]]])
        sigma = np.array([[[0.229]], [[0.224]], [[0.225]]])

        show_ref_img, show_prev_img, show_curr_img = [
            img.cpu().numpy()[0] * sigma + mean
            for img in [ref_imgs, prev_imgs, curr_imgs]
        ]

        show_gt, show_prev_gt, show_ref_gt, show_preds_s = [
            label.cpu()[0].squeeze(0).numpy()
            for label in [curr_labels, prev_labels, ref_labels, curr_pred]
        ]

        show_gtf, show_prev_gtf, show_ref_gtf, show_preds_sf = [
            label2colormap(label).transpose((2, 0, 1))
            for label in [show_gt, show_prev_gt, show_ref_gt, show_preds_s]
        ]

        if cfg.TRAIN_IMG_LOG or cfg.TRAIN_TBLOG:

            show_ref_img = masked_image(show_ref_img, show_ref_gtf,
                                        show_ref_gt)
            if cfg.TRAIN_IMG_LOG:
                save_image(
                    show_ref_img,
                    os.path.join(cfg.DIR_IMG_LOG,
                                 '%06d_ref_img.jpeg' % (step)))

            show_prev_img = masked_image(show_prev_img, show_prev_gtf,
                                         show_prev_gt)
            if cfg.TRAIN_IMG_LOG:
                save_image(
                    show_prev_img,
                    os.path.join(cfg.DIR_IMG_LOG,
                                 '%06d_prev_img.jpeg' % (step)))

            show_img_pred = masked_image(show_curr_img, show_preds_sf,
                                         show_preds_s)
            if cfg.TRAIN_IMG_LOG:
                save_image(
                    show_img_pred,
                    os.path.join(cfg.DIR_IMG_LOG,
                                 '%06d_prediction.jpeg' % (step)))

            show_curr_img = masked_image(show_curr_img, show_gtf, show_gt)
            if cfg.TRAIN_IMG_LOG:
                save_image(
                    show_curr_img,
                    os.path.join(cfg.DIR_IMG_LOG,
                                 '%06d_groundtruth.jpeg' % (step)))

            if cfg.TRAIN_TBLOG:
                for seq_step, running_loss, running_iou in zip(
                        range(len(running_losses)), running_losses,
                        running_ious):
                    self.tblogger.add_scalar('S{}/Loss'.format(seq_step),
                                             running_loss.avg, step)
                    self.tblogger.add_scalar('S{}/IoU'.format(seq_step),
                                             running_iou.avg, step)

                self.tblogger.add_scalar('LR', now_lr, step)
                self.tblogger.add_image('Ref/Image', show_ref_img, step)
                self.tblogger.add_image('Ref/GT', show_ref_gtf, step)

                self.tblogger.add_image('Prev/Image', show_prev_img, step)
                self.tblogger.add_image('Prev/GT', show_prev_gtf, step)

                self.tblogger.add_image('Curr/Image_GT', show_curr_img, step)
                self.tblogger.add_image('Curr/Image_Pred', show_img_pred, step)

                self.tblogger.add_image('Curr/Mask_GT', show_gtf, step)
                self.tblogger.add_image('Curr/Mask_Pred', show_preds_sf, step)

                for key in boards['image'].keys():
                    tmp = boards['image'][key].cpu().numpy()
                    self.tblogger.add_image('S{}/' + key, tmp, step)
                for key in boards['scalar'].keys():
                    tmp = boards['scalar'][key].cpu().numpy()
                    self.tblogger.add_scalar('S{}/' + key, tmp, step)

                self.tblogger.flush()

        del (boards)
