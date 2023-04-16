from .default import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'R50_AOTL'

        self.MODEL_ENCODER = 'resnet50'
        self.MODEL_ENCODER_PRETRAIN = './pretrain_models/resnet50-0676ba61.pth'  # https://download.pytorch.org/models/resnet50-0676ba61.pth
        self.MODEL_ENCODER_DIM = [256, 512, 1024, 1024]  # 4x, 8x, 16x, 16x
        self.MODEL_LSTT_NUM = 3

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 5