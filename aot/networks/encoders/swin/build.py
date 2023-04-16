# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer


def build_swin_model(model_type, freeze_at=0):
    if model_type == 'swin_base':
        model = SwinTransformer(embed_dim=128,
                                depths=[2, 2, 18, 2],
                                num_heads=[4, 8, 16, 32],
                                window_size=7,
                                drop_path_rate=0.3,
                                out_indices=(0, 1, 2),
                                ape=False,
                                patch_norm=True,
                                frozen_stages=freeze_at,
                                use_checkpoint=False)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
