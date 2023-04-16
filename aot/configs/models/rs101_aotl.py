from .default import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'R101_AOTL'

        self.MODEL_ENCODER = 'resnest101'
        self.MODEL_ENCODER_PRETRAIN = './pretrain_models/resnest101-22405ba7.pth'  # https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest101-22405ba7.pth
        self.MODEL_ENCODER_DIM = [256, 512, 1024, 1024]  # 4x, 8x, 16x, 16x
        self.MODEL_LSTT_NUM = 3

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 5