
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import copy
import os
import time
import json
import logging
import torchvision


def loss_fn(gen_x, x, mu, logvar):
    """
    gen_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = nn.BCELoss(reduction="sum")(gen_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # print(KLD_element.shape)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return (BCE + KLD) / x.shape[0], BCE / x.shape[0], KLD / x.shape[0]


def onehot(idx, n):
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

class VAE(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(512, self.latent_dim)
        self.log_std_head = nn.Linear(512, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )


    def encode(self, x):
        x = x.view(-1, 784)
        feature = self.backbone(x)
        return self.mean_head(feature), self.log_std_head(feature)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)

        return self.decode(z), mu, logvar
    def sample(self, num_samples, device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(device)

        samples = self.decode(z)
        return samples


class CVAE(nn.Module):
    def __init__(self, input_size, latent_dim, num_class):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_size + num_class, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(512, self.latent_dim)
        self.log_std_head = nn.Linear(512, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim + num_class, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )


    def encode(self, x, c):
        c = onehot(c, 10)
        x = x.view(-1, 784)
        x = torch.cat([x, c], axis = 1)
        feature = self.backbone(x)
        return self.mean_head(feature), self.log_std_head(feature)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, c):
        c = onehot(c, 10)
        x = torch.cat([z, c], axis = 1)
        return self.decoder(x)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparametrize(mu, logvar)

        return self.decode(z, c), mu, logvar
    def sample(self, num_samples, device, c):
        """
        """
        c = torch.tensor([c] * num_samples).to(device)
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(device)

        samples = self.decode(z, c)
        return samples