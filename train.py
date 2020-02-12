import argparse
import glob
import logging
import os
import re

import natsort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from import_model import *


def args_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100000,
                        help='epoch')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='batch_size')
    parser.add_argument('--lr', type=int, default=0.0001,
                        help='learning rate')
    parser.add_argument('--continue_training', type=bool, default=True,
                        help='if continue training')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='print log every n epoch')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='save checkpoint every n epoch')

    parser.add_argument('--signal_len', type=int, default=60,
                        help='raw imput signal length, like tensor.shape==(N, 1, 60) is 60')
    parser.add_argument('--latent_dim', type=int, default=8,
                        help='latent dimension')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='if use gpu')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='dataloader workers')
    parser.add_argument('--test_batch_size', type=int, default=100,
                        help='test batch_size')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    logging.info(args)
    return args


def logging_set():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler("log_training.log", mode='w')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[file_handler, stream_handler]
    )


def main():
    os.makedirs('model_saved', exist_ok=True)
    torch.manual_seed(0)
    args = args_gen()

    rri_fp = './rris_tensor_norm.pt'
    rris_tensor = torch.load(rri_fp)

    all_set = TensorDataset(rris_tensor)
    split_shape = (int(rris_tensor.shape[0] * 0.8), rris_tensor.shape[0] - int(rris_tensor.shape[0] * 0.8))
    train_set, test_set = random_split(all_set, split_shape)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    logging.info(f"training set length: {len(train_set)}")

    model = AutoEncoder(args).to(args.device)
    logging.info(f'model structure:\n {model}')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    continue_epoch = 0
    if args.continue_training:
        try:
            model_fp = natsort.natsorted(glob.glob('model_saved/model*.pt'))[-1]
            optimizer_fp = natsort.natsorted(glob.glob('model_saved/optimizer*.pt'))[-1]
            logging.info(f"load model file: '{model_fp}")
            logging.info(f"load optimizer file: '{optimizer_fp}'")
            model_epoch = [int(x) for x in re.findall(r'\d+', model_fp)][0]
            optimizer_epoch = [int(x) for x in re.findall(r'\d+', optimizer_fp)][0]
            if model_epoch == optimizer_epoch:
                model.load_state_dict(torch.load(model_fp, map_location=args.device))
                optimizer.load_state_dict(torch.load(optimizer_fp, map_location=args.device))
                continue_epoch = model_epoch
                logging.info(f'continue training from epoch: {continue_epoch}')
        except:
            logging.exception('[ ERROR ] load model checkpoint failed')

    model.train()
    for epoch in range(continue_epoch, args.epoch):
        for step, (x,) in enumerate(train_loader):
            x = x.to(args.device)
            encoded, decoded = model(x)

            loss = loss_func(decoded, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % args.log_interval == 0:
            total_step = epoch * len(train_loader) + step
            logging.info((f"epoch: {epoch}", f"total_step: {total_step}", f"loss: {loss.data.cpu().numpy():.4f}"))
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), f'model_saved/model_{epoch}.pt')
            torch.save(optimizer.state_dict(), f'model_saved/optimizer_{epoch}.pt')
    torch.save(model.state_dict(), 'model_saved/model_fin.pt')
    torch.save(optimizer.state_dict(), 'model_saved/optimizer_fin.pt')


if __name__ == "__main__":
    logging_set()
    main()
