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
    parser.add_argument('--test_batch_size', type=int, default=10,
                        help='test batch_size')
    parser.add_argument('--signal_len', type=int, default=60,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=8,
                        help='latent dim')
    parser.add_argument('--use_gpu', type=bool, default=False,
                        help='if use gpu')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print(args)
    return args


def main():
    torch.manual_seed(0)
    args = args_gen()

    rri_fp = './rris_tensor_norm.pt'
    rris_tensor = torch.load(rri_fp)
    all_set = TensorDataset(rris_tensor)
    split_shape = (int(rris_tensor.shape[0] * 0.8), rris_tensor.shape[0] - int(rris_tensor.shape[0] * 0.8))
    train_set, test_set = random_split(all_set, split_shape)

    model = AutoEncoder(args).to(args.device)
    # print(f'model structure:\n {model}')
    model_fp = natsort.natsorted(glob.glob('model_saved/model*.pt'))[-1]
    print(f'using model file: {model_fp}')
    model.load_state_dict(torch.load(model_fp, map_location=torch.device('cpu')))
    model.eval()

    # disable manual seed
    torch.seed()

    test_loader = DataLoader(test_set, args.test_batch_size, shuffle=True)
    with torch.no_grad():
        for test_x, in test_loader:
            _, test_y = model(test_x)

            plt.subplots()
            plt.plot(test_x[0][0], '-b.', label='raw_input')
            plt.plot(test_y[0][0], '-ro', label='predicted')

            plt.title(model_fp)
            plt.legend()
            plt.show()


if __name__ == "__main__":
    main()
