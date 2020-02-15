import argparse
import glob
import os
import natsort

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from import_model import *


def args_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, help='test file path',
                        default='nsr2db_rris_tensor_norm_lim.pt')
    parser.add_argument('--test_batch_size', type=int, default=20,
                        help='test batch_size')
    parser.add_argument('--signal_len', type=int, default=60,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=8,
                        help='latent dim')
    parser.add_argument('--auto_close_fig', type=bool, default=False,
                        help='if auto close figure while viewing results')
    parser.add_argument('--use_gpu', type=bool, default=False,
                        help='if use gpu')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print(args)
    return args


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
                    ax[idx].set_title(model_fp)
                    ax[idx].legend()
            plt.tight_layout()
            if args.auto_close_fig:
                plt.show(block=False)
                plt.pause(5)
                plt.close()
            else:
                plt.show()


def main():
    torch.manual_seed(0)
    args = args_gen()

    rris_tensor = torch.load(args.test_file).to(args.device)
    print(f'loaded file: {args.test_file}')
    all_set = TensorDataset(rris_tensor)
    split_shape = (int(rris_tensor.shape[0] * 0.8), rris_tensor.shape[0] - int(rris_tensor.shape[0] * 0.8))
    train_set, test_set = random_split(all_set, split_shape)
    test_loader = DataLoader(test_set, args.test_batch_size, shuffle=True)
    apply_model(test_loader, args)


if __name__ == "__main__":
    main()
