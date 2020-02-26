import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.features = nn.Sequential(
            nn.ConvTranspose1d(1, 64, 5),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((args.latent_dim, args.latent_dim))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            nn.Upsample(16),
            nn.Conv1d(args.latent_dim, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Upsample(32),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(args.signal_len),
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 1, kernel_size=9, padding=4),
        )

    def forward(self, x):
        x = self.net(x)
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
