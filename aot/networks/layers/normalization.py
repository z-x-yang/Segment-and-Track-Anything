import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """
    def __init__(self, n, epsilon=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n) - epsilon)
        self.epsilon = epsilon

    def forward(self, x):
        """
        Refer to Detectron2 (https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/layers/batch_norm.py)
        """
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.epsilon).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.epsilon,
            )
