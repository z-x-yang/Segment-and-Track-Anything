from networks.engines.aot_engine import AOTEngine, AOTInferEngine
from networks.engines.deaot_engine import DeAOTEngine, DeAOTInferEngine


def build_engine(name, phase='train', **kwargs):
    if name == 'aotengine':
        if phase == 'train':
            return AOTEngine(**kwargs)
        elif phase == 'eval':
            return AOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    elif name == 'deaotengine':
        if phase == 'train':
            return DeAOTEngine(**kwargs)
        elif phase == 'eval':
            return DeAOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
