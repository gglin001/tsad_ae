import glob

import matplotlib.pyplot as plt
import natsort
import numpy as np
import torch
from scipy.stats import pearsonr
from import_model import *

from train import parse_args

args = parse_args()
model = AutoEncoder(args).to(args.device)
checkpoint_fps = glob.glob('model_saved/checkpoint_*.pt')
checkpoint_fp = checkpoint_fps[0] if len(checkpoint_fps) == 0 else natsort.natsorted(checkpoint_fps)[-1]
# checkpoint_fp = natsort.natsorted(checkpoint_fps)[-2]
print(f'using model file: {checkpoint_fp}')
checkpoint = torch.load(checkpoint_fp, map_location=args.device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def apply_model_loader(loader, args):
    with torch.no_grad():
        for x, in loader:
            _, y_p = model(x)
            y_t = x.reshape(x.shape[0], -1)
            y_p = y_p.reshape(y_p.shape[0], -1)

            # _view_data(y_t, y_p, args)
            return _apply_thld(y_t, y_p)


def apply_model(x):
    x = torch.as_tensor(x).float()
    with torch.no_grad():
        _, y_t = model(x)
        y_p = x.reshape(x.shape[0], -1)
        y_t = y_t.reshape(y_t.shape[0], -1)

    # _view_data(y_t, y_p)
    return _apply_thld(y_t, y_p)


def _apply_thld(y_t, y_p, l1_thld=2, l2_thld=0.3, corr_thld=0.6):
    minus = y_t - y_p
    l1 = np.asarray([np.linalg.norm(x, ord=1) for x in minus])
    l2 = np.asarray([np.linalg.norm(x, ord=2) for x in minus])
    corr = np.asarray([pearsonr(p, t)[0] for p, t in zip(y_p, y_t)])

    out = np.zeros((len(y_t), ))

    l1_ones_idxs = np.where(l1 > l1_thld)[0]
    l2_ones_idxs = np.where(l2 > l2_thld)[0]
    corr_ones_idxs = np.where(corr < corr_thld)[0]

    one_idxs = np.intersect1d(l1_ones_idxs, l2_ones_idxs)
    one_idxs = np.intersect1d(one_idxs, corr_ones_idxs)

    print(f"y_t len: {len(y_t)}, one_idx len: {len(one_idxs)}, percent: {len(one_idxs)/len(y_t)}")

    out[one_idxs] = 1
    return out


def _view_data(y_t, y_p, args):
    minus = y_t - y_p
    l1 = np.asarray([np.linalg.norm(x, ord=1) for x in minus])
    l1 = l1[l1 < 10]
    plt.subplots()
    plt.hist(l1, bins=30, ec='k')
    plt.title('l1')

    l2 = np.asarray([np.linalg.norm(x, ord=2) for x in minus])
    # l2 = l2[l2 < 10]
    plt.subplots()
    plt.hist(l2, bins=30, ec='k')
    plt.title('l2')

    corr_pvalue = np.asarray([pearsonr(p, t) for p, t in zip(y_p, y_t)])
    corr = corr_pvalue[:, 0]
    plt.subplots()
    plt.hist(corr, bins=30, ec='k')
    plt.title('corr')

    pvalue = corr_pvalue[:, 1]
    plt.subplots()
    plt.hist(pvalue, bins=30, ec='k')
    plt.title('pvalue')
    plt.show()

    nrows = 4
    ncols = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))
    ax = ax.ravel()
    for idx in range(min(nrows * ncols, args.test_batch_size)):
        ax[idx].plot(y_t[idx], '-b.', label='raw_input')
        ax[idx].plot(y_p[idx], '-r.', label='predicted')

        # ax[idx].set_ylim([0, 1])

        if idx == 0:
            ax[idx].legend()

    plt.tight_layout()
    plt.suptitle(f'data:{args.test_file}\nmodel:{checkpoint_fp}',
                 **{'color': 'm', 'bbox': dict(facecolor='w', alpha=0.7)})
    plt.show()
