import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from apply_model import apply_model_loader as apply_model
from import_model import *
from utils import view_res


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = torch.device('cpu')
    args.test_batch_size = 20
    # args.test_file = 'normal.npy'
    args.test_file = 'anormal.npy'

    rris_np = np.load(args.test_file)
    all_set = TensorDataset(torch.as_tensor(rris_np).float())
    train_len = int(len(all_set) * 0.8)
    val_len = int(len(all_set) * 0.1)
    test_len = len(all_set) - train_len - val_len
    split_shape = (train_len, val_len, test_len)
    torch.manual_seed(0)
    train_set, val_set, test_set = random_split(all_set, split_shape)
    torch.seed()  # disable manual seed

    loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True)
    apply_model(loader, args)


if __name__ == "__main__":
    main()
