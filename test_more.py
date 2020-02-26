import argparse
import glob
import os
import natsort

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from import_model import *
from test import apply_model
from argparse_set import args_gen


def main():
    args = args_gen()
    args.test_file = 'nsr2db_rris_tensor_norm_lim.pt'

    rris_tensor = torch.load(args.test_file).to(args.device)
    print(f'loaded file: {args.test_file}')
    test_set = TensorDataset(rris_tensor)
    test_loader = DataLoader(test_set, args.test_batch_size, shuffle=True)
    apply_model(test_loader, args)


if __name__ == "__main__":
    main()
