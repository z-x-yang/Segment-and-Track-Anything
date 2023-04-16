from networks.models.aot import AOT
from networks.models.deaot import DeAOT


def build_vos_model(name, cfg, **kwargs):
    if name == 'aot':
        return AOT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    elif name == 'deaot':
        return DeAOT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    else:
        raise NotImplementedError
