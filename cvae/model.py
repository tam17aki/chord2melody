# -*- coding: utf-8 -*-
"""Model definition for chord2melody.

Copyright (C) 2022 Rui Konuma
Copyright (C) 2023 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
from torch import nn


class Encoder(nn.Module):
    """RNN(LSTM)-based Encoder class."""

    def __init__(self, cfg):
        """Initialize class.

        cfg (Config) : configures for encoder.
        """
        super().__init__()
        self.cfg = cfg
        self.bidirectional = cfg.bidirectional
        self.n_fc_layers = cfg.n_fc_layers
        self.embedding = nn.Embedding(cfg.input_dim, cfg.emb_dim)
        self.rnn = nn.LSTM(
            cfg.emb_dim + cfg.condition_dim,
            cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )
        self.fc_layers = nn.ModuleList([])
        for layer in range(cfg.n_fc_layers):
            if layer == 0:
                in_channels = (
                    2 * cfg.hidden_dim if cfg.bidirectional else cfg.hidden_dim
                )
            else:
                in_channels = cfg.hidden_dim
            self.fc_layers += [nn.Linear(in_channels, cfg.hidden_dim)]
        self.fc_layers += [
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        ]
        self.activation = nn.ReLU()

    def forward(self, melody, condition):
        """Forward propagation.

        Args:
            melody (Tensor) : sequence of note numbers
            condition (Tensor) : sequence of chord vectors
        """
        embedded = self.embedding(melody)
        source = torch.cat([embedded, condition], axis=-1)
        _, (hidden, _) = self.rnn(source)
        if self.bidirectional:
            # hidden: [2, N, D] for bidirectional
            hidden = torch.cat(hidden[:2, :, :], dim=-1)  # forward and backward
        else:
            hidden = hidden[0]  # only forward
        for i in range(self.n_fc_layers):
            hidden = self.activation(self.fc_layers[i](hidden))
        mean = self.fc_layers[-2](hidden)
        logvar = self.fc_layers[-1](hidden)
        return hidden, mean, logvar


class Decoder(nn.Module):
    """RNN(LSTM)-based Decoder class."""

    def __init__(self, cfg):
        """Initialize class.

        cfg (Config) : configures for encoder.
        """
        super().__init__()
        self.reverse_latent = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.n_layers = cfg.n_layers
        self.n_fc_layers = cfg.n_fc_layers
        self.rnn = nn.LSTM(
            cfg.hidden_dim + cfg.condition_dim,
            cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )
        self.fc_layers = nn.ModuleList([])
        self.fc_layers += [
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.n_fc_layers)
        ]
        self.fc_out = nn.Linear(
            2 * cfg.hidden_dim if cfg.bidirectional else cfg.hidden_dim, cfg.output_dim
        )
        self.activation = nn.ReLU()

    def forward(self, latent, condition):
        """Forward propagation.

        Args:
            latent (Tensor) : latent vectors of VAE
            condition (Tensor) : sequence of chord vectors

        Returns:
            outputs (Tensor) : sequence of one-hot vectors (melody)
        """
        hidden = self.reverse_latent(latent)
        for i in range(self.n_fc_layers):
            hidden = self.activation(self.fc_layers[i](hidden))
        seq_len = condition.shape[1]
        hidden = hidden.unsqueeze(1)
        hidden = hidden.repeat(1, seq_len, 1)
        hidden = torch.cat([hidden, condition], dim=-1)
        hidden, _ = self.rnn(hidden)
        outputs = self.fc_out(hidden)
        return outputs


class MelodyComposer(nn.Module):
    """MelodyComposer class for ad-lib melody composition.

    The architecture is an LSTM-CVAE (a conditional VAE).
    """

    def __init__(self, config, device):
        """Initialize class.

        config (Config) : configures for encoder/decoder model.
        """
        super().__init__()
        self.encoder = Encoder(config.encoder).to(device)
        self.decoder = Decoder(config.decoder).to(device)
        assert (
            config.encoder.latent_dim == config.decoder.latent_dim
        ), "latant_dim must be matched between encoder and decoder."

    def encode(self, inputs, condition):
        """Drive encoder to get latent vectors."""
        hidden, mean, logvar = self.encoder(inputs, condition)
        return hidden, mean, logvar

    def decode(self, latent, condition):
        """Drive decoder to reconstruct input from latent vectors."""
        reconst = self.decoder(latent, condition)
        return reconst

    def forward(self, inputs, condition):
        """Forward propagation.

        Args:
            inputs (Tensor) : sequence of mixed-hot vectors (melody + chord)
            condition (Tensor) : sequence of chord vectors

        Returns:
            reconst (Tensor) : reconstructed inputs (melody)
            mean (Tensor) : mean vector of VAE
            logvar (Tensor) : log variance vector an vec (diag. cov.)tor of VAE
        """
        _, mean, logvar = self.encode(inputs, condition)
        latent = self.reparameterization(mean, logvar)
        reconst = self.decode(latent, condition)
        return reconst, mean, logvar

    def reparameterization(self, mean, logvar):
        """Sample latent vector from inputs via reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mean + eps * std  # reparameterization trick
        return latent


def get_model(config, device):
    """Instantiate model."""
    model = MelodyComposer(config.model, device)
    return model
