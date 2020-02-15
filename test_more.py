import argparse
import glob
import os
import natsort

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from import_model import *
from test import apply_model, args_gen


def args_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, help='test file path',
                        default='ltafdb_rris_tensor_norm.pt')
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


def main():
    args = args_gen()

    rris_tensor = torch.load(args.test_file).to(args.device)
    print(f'loaded file: {args.test_file}')
    test_set = TensorDataset(rris_tensor)
    test_loader = DataLoader(test_set, args.test_batch_size, shuffle=True)
    apply_model(test_loader, args)


if __name__ == "__main__":
    main()
