import argparse
import torch

def args_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100000,
                        help='epoch')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='batch_size')
    parser.add_argument('--lr', type=int, default=0.0001,
                        help='learning rate')

    parser.add_argument('--log_interval', type=int, default=10,
                        help='print log every n epoch')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='save checkpoint every n epoch')

    parser.add_argument('--signal_len', type=int, default=60,
                        help='raw imput signal length, like tensor.shape==(N, 1, 60) is 60')
    parser.add_argument('--latent_dim', type=int, default=6,
                        help='latent dimension')

    parser.add_argument('--eval_val', type=bool, default=True,
                        help='if eval validation set (True | False)')
    parser.add_argument('--continue_training', type=bool, default=False,
                        help='if continue training (True | False)')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='if use gpu (True | False)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='dataloader workers')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    return args
