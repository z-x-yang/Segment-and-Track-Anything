import os
from .default import DefaultModelConfig

class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'AOTL'

        self.MODEL_LSTT_NUM = 3

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 5