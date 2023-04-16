from .default import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'SwinB_AOTL'

        self.MODEL_ENCODER = 'swin_base'
        self.MODEL_ENCODER_PRETRAIN = './pretrain_models/swin_base_patch4_window7_224_22k.pth'  # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
        self.MODEL_ALIGN_CORNERS = False
        self.MODEL_ENCODER_DIM = [128, 256, 512, 512]  # 4x, 8x, 16x, 16x
        self.MODEL_LSTT_NUM = 3

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 5