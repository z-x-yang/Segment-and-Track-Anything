from .default_deaot import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'SwinB_DeAOTL'

        self.MODEL_ENCODER = 'swin_base'
        self.MODEL_ALIGN_CORNERS = False
        self.MODEL_ENCODER_DIM = [128, 256, 512, 512]  # 4x, 8x, 16x, 16x

        self.MODEL_LSTT_NUM = 3

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 5