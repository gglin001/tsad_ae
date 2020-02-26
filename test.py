import argparse
import glob
import os
import natsort

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from import_model import *
from argparse_set import args_gen


def apply_model(test_loader, args):
    model = AutoEncoder(args).to(args.device)
    # print(f'model structure:\n {model}')
    model_fp = natsort.natsorted(glob.glob('model_saved/model*.pt'))[-1]
    print(f'using model file: {model_fp}')
    model.load_state_dict(torch.load(model_fp, map_location=torch.device('cpu')))
    model.eval()

    # disable manual seed
    torch.seed()

    with torch.no_grad():
        for test_x, in test_loader:
            _, test_y = model(test_x)
            test_x = test_x.reshape(test_x.shape[0], -1)
            test_y = test_y.reshape(test_y.shape[0], -1)

            nrows = 4
            ncols = 5
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))
            ax = ax.ravel()
            for idx in range(min(nrows * ncols, args.test_batch_size)):
                ax[idx].plot(test_x[idx], '-b.', label='raw_input')
                ax[idx].plot(test_y[idx], '-ro', label='predicted')
                if idx == 0:
                    ax[idx].legend()
            plt.tight_layout()
            plt.suptitle(f'data:{args.test_file}\nmodel:{model_fp}',
                         **{'color': 'm', 'bbox': dict(facecolor='w', alpha=0.7)})
            plt.show()


def main():
    args = args_gen()
    args.device = torch.device('cpu')
    args.test_file = 'nsr2db_rris_tensor_norm_lim.pt'

    rris_tensor = torch.load(args.test_file).to(args.device)
    print(f'loaded file: {args.test_file}')
    all_set = TensorDataset(rris_tensor)
    split_shape = (int(rris_tensor.shape[0] * 0.8), rris_tensor.shape[0] - int(rris_tensor.shape[0] * 0.8))

    torch.manual_seed(0)
    train_set, test_set = random_split(all_set, split_shape)
    test_loader = DataLoader(test_set, args.test_batch_size, shuffle=True)
    apply_model(test_loader, args)


if __name__ == "__main__":
    main()
