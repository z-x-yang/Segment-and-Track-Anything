from .default import DefaultModelConfig as BaseConfig


class DefaultModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'DeAOTDefault'

        self.MODEL_VOS = 'deaot'
        self.MODEL_ENGINE = 'deaotengine'

        self.MODEL_DECODER_INTERMEDIATE_LSTT = False

        self.MODEL_SELF_HEADS = 1
        self.MODEL_ATT_HEADS = 1

        self.TRAIN_AUG_TYPE = 'v2'
