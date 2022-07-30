import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def check_dir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def onehot_encoding(logits, num):
    onehot = torch.zeros(len(logits), num, device=logits.device).int()
    return onehot.scatter_(1, logits.reshape(-1, 1), 1)

def render_pc(ax, pc, mask, figsize=(8, 8)):
    ax.view_init(elev=20, azim=60)
    x = pc[:, 0]
    y = pc[:, 2]
    z = pc[:, 1]
    # ax.scatter(x[mask], y[mask], z[mask], marker='.', color='#FF0000')
    ax.scatter(x[mask], y[mask], z[mask], marker='.', color='#00FF00')
    ax.scatter(x[~mask], y[~mask], z[~mask], marker='.', color='#00FF00')
    miv = np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space.
    mav = np.max([np.max(x), np.max(y), np.max(z)])
    ax.set_xlim(miv, mav)
    ax.set_ylim(miv, mav)
    ax.set_zlim(miv, mav)
    plt.tight_layout()

def set_index(pred_np):
    return np.c_[np.arange(pred_np.shape[1]+1)-1, np.r_[np.arange(pred_np.shape[0]).reshape(1, -1), pred_np]]
