import seaborn as sns
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
from model import loss_fn as loss_func
from model import CVAE
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch import Tensor
from tqdm import tqdm
from model import VAE
from model import CVAE
import pandas as pd
np.random.seed(0)
torch.manual_seed(0)# 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed_all(0)# 为所有的GPU设置种子，以使得结果是确定的

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def get_dist(model, optimizer, lr_scheduler, dataloaders, device, condition):
    if mutil_gpu:
        model = model.module
    dist = {"x1":np.array([]), "x2":np.array([]), "y":np.array([])}
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloaders['train']):
            x = x.to(device=device)
            y = y.to(device=device)
            if condition:
                mu, logvar = model.encode(x, y)
            else:
                mu, logvar = model.encode(x)
            mu = mu.cpu().numpy()
            y = y.cpu().numpy()
            dist["x1"] = np.concatenate([dist["x1"], mu[:, 0]])
            dist["x2"] = np.concatenate([dist["x2"], mu[:, 1]])
            dist["y"] = np.concatenate([dist["y"], y])
    return dist


def load_model(file_path, model, optimizer = None, lr_scheduler = None):
    state_dicts = torch.load(file_path, map_location="cpu")
    model.load_state_dict(state_dicts["model"])
    if optimizer:
        optimizer.load_state_dict(state_dicts["optimizer"])
    if lr_scheduler:
        lr_scheduler.load_state_dict(state_dicts["scheduler"])

task_name = "CVAE_on_MNIST"
model_name = "CVAE"
optimizer_name = 'Adam'
lr = 0.01
weight_decay = 1e-4
step_size = 100
gamma = 0.5
batch_size = 256
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
load_checkpoint = True
mutil_gpu = False
device_ids = ["cuda:1", "cuda:2"]
condition = False



if __name__ == "__main__":
    mean, std = 0, 1

    if condition:
        model = CVAE(input_size = 784, latent_dim = 2, num_class = 10)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) 
        load_model("model_checkpoint_CVAE/check_point.pkl", model, optimizer, lr_scheduler)
    else:
        model = VAE(input_size = 784, latent_dim = 2)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) 
        load_model("model_checkpoint/check_point.pkl", model, optimizer, lr_scheduler)

    if mutil_gpu:
        model = nn.DataParallel(model, device_ids, device)
    model = model.to(device=device)
    mnist = torchvision.datasets.MNIST(root = "mnist", train=True, download=True, transform=transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean, std)
            ]
        ))
    dataLoaders = {"train":torch.utils.data.DataLoader(mnist,
            batch_size=batch_size, shuffle=True, num_workers= 0, pin_memory=True, drop_last=False)}

    dist = get_dist(model, optimizer, lr_scheduler, dataLoaders, device, condition)
    # print(dist)
    df = pd.DataFrame.from_dict(dist, orient='columns')
    g = sns.lmplot(x='x1', y='x2', hue='y', data=df.groupby('y').head(100), fit_reg=False, legend=True)
    g.savefig("images/{}_dist.png".format("VAE" if not condition else "CVAE"),dpi=300)