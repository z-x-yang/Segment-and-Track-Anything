from __future__ import absolute_import


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, momentum=0.999):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.long_count = 0
        self.momentum = momentum
        self.moving_avg = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.long_count == 0:
            self.moving_avg = val
        else:
            momentum = min(self.momentum, 1. - 1. / self.long_count)
            self.moving_avg = self.moving_avg * momentum + val * (1 - momentum)
        self.val = val
        self.sum += val * n
        self.count += n
        self.long_count += n
        self.avg = self.sum / self.count
