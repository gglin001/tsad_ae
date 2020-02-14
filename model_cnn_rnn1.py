import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.cnn_1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.rnn_1 = nn.LSTM(args.signal_len, 64, 8, bidirectional=True)
        self.pool_1 = nn.MaxPool1d(2)
        self.cnn_2 = nn.Conv1d(32, 16, kernel_size=5, padding=2)
        self.rnn_2 = nn.LSTM(64, 16, 8, bidirectional=True)
        self.pool_2 = nn.MaxPool1d(2)

        self.avgpool = nn.AdaptiveAvgPool2d((args.latent_dim, args.latent_dim))

    def forward(self, x):
        x = self.cnn_1(x)
        x = torch.relu(x)
        x, _ = self.rnn_1(x)
        x = torch.relu(x)
        x = self.pool_1(x)
        x = self.cnn_2(x)
        x = torch.relu(x)
        x, _ = self.rnn_2(x)
        x = self.pool_2(x)

        x = self.avgpool(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.rnn_1 = nn.LSTM(args.latent_dim, 32, 8, bidirectional=True)
        self.cnn_1 = nn.Conv1d(args.latent_dim, 16, kernel_size=3, padding=1)
        self.pool_1 = nn.MaxPool1d(2)

        self.rnn_2 = nn.LSTM(32, args.signal_len, 8, bidirectional=True)
        self.pool_2 = nn.MaxPool1d(2)
        self.cnn_2 = nn.Conv1d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x, _ = self.rnn_1(x)
        x = torch.relu(x)
        x = self.cnn_1(x)
        x = torch.relu(x)
        x = self.pool_1(x)

        x, _ = self.rnn_2(x)
        x = self.pool_2(x)
        x = self.cnn_2(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


if __name__ == "__main__":
    from test import args_gen

    args = args_gen()
    encoder = Encoder(args)
    decoder = Decoder(args)

    input_ = torch.randn(20, 1, 60)
    print(input_.shape)

    output = encoder(input_)
    print(output.shape)

    try:
        output = decoder(output)
        print(output.shape)
    except:
        pass
