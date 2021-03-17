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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
np.random.seed(0)
torch.manual_seed(0)# 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed_all(0)# 为所有的GPU设置种子，以使得结果是确定的

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def sample(model, device, num):
    print("start sampling...")
    logging.warning("start sampling...")
    model.eval()
    if mutil_gpu:
        model = model.module
    with torch.no_grad():
    
        fig = plt.figure(figsize=(num, 10))
        gs = gridspec.GridSpec(10, num)
        gs.update(wspace=0.05, hspace=0.05)

        for i in range(10):
            print("sampling {}...".format(i + 1))
            logging.warning("sampling {}...".format(i + 1))
            samples = model.sample(num, device, i).cpu().numpy()
            for j, sample in enumerate(samples):
                ax = plt.subplot(gs[i, j])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28, 28) * 255, cmap='Greys_r')

        if not os.path.exists('out_CVAE/'):
            os.makedirs('out_CVAE/')

        plt.savefig('out_CVAE/{}.png'.format("sample"), bbox_inches='tight')

        plt.close(fig)
def load_model(file_path, model, optimizer = None, lr_scheduler = None):
    state_dicts = torch.load(file_path, map_location="cpu")
    model.load_state_dict(state_dicts["model"])
    if optimizer:
        optimizer.load_state_dict(state_dicts["optimizer"])
    if lr_scheduler:
        lr_scheduler.load_state_dict(state_dicts["scheduler"])

if __name__ == "__main__":
    task_name = "eval CVAE"
    logging.basicConfig(filename = "{}.log".format(task_name))
    mutil_gpu = False
    model = CVAE(input_size = 784, latent_dim = 2, num_class = 10)
    optimizer = optim.Adam(model.parameters(), 0.001)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) 
    if True:
        load_model("model_checkpoint_CVAE/check_point.pkl", model, optimizer, lr_scheduler)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    sample(model, device, 4)
