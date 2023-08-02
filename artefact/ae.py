import torch.nn as nn
import numpy as np
from torch import nn, optim


def build_net(dims, reluout=True, dropout: float | None = None, leaky=False):
    ret = []
    n = len(dims[:-1])
    for i, (n1, n2) in enumerate(zip(dims[:-1], dims[1:])):
        ret.append(nn.Linear(n1, n2))
        if dropout:
            ret.append(nn.Dropout(dropout))
        if i < n - 1 or reluout:
            # ret.append(nn.ReLU())
            if leaky:
                ret.append(nn.LeakyReLU())
            else:
                ret.append(nn.ReLU())
    return ret


def build_encoder(dims, reluout=True, dropout: float | None = None, leaky=False):
    return build_net(dims, reluout, dropout, leaky)


def build_decoder(dims, dropout: float | None = None, leaky=False):
    # return build_net(dims, dropout=dropout)[:-1] + [nn.Sigmoid()]
    return build_net(dims, dropout=dropout, leaky=leaky)[:-1] + [nn.Tanh()]


class Autoencoder(nn.Module):
    def __init__(self, layers, reluout=True, dropout: float | None = None, leaky=False):
        """Build a new encoder using the architecture specified with
        [arch_encoder] and [arch_decoder].
        """

        super().__init__()
        if dropout:
            print("using dropout!")
        if layers[0] != layers[-1]:
            arch_encoder = layers
            arch_decoder = tuple(reversed(layers))
        else:
            latent_index = np.argmin(layers)
            arch_encoder = layers[: i + 1]
            arch_decoder = layers[i:]

        self.encoder = nn.Sequential(*build_encoder(arch_encoder, reluout, dropout, leaky))

        arch_decoder = list(reversed(arch_encoder)) if arch_decoder is None else arch_decoder
        decode = build_decoder(arch_decoder, dropout, leaky)
        if not reluout:
            decode = [nn.ReLU()] + decode
        self.decoder = nn.Sequential(*decode)
        print("encoder", self.encoder)
        print("decoder", self.decoder)

    def forward(self, x, **kwargs):
        lat = self.encoder(x)
        output = self.decoder(lat)
        return lat, output
