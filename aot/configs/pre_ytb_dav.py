import os
from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='AOTT'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'PRE_YTB_DAV'

        self.init_dir()

        self.DATASETS = ['youtubevos', 'davis2017']

        pretrain_stage = 'PRE'
        pretrain_ckpt = 'save_step_100000.pth'
        self.PRETRAIN_FULL = True  # if False, load encoder only
        self.PRETRAIN_MODEL = os.path.join(self.DIR_ROOT, 'result',
                                           self.EXP_NAME, pretrain_stage,
                                           'ema_ckpt', pretrain_ckpt)
