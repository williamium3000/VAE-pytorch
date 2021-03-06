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
from model import VAE
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

def sample(model, device, epoch):
    model.eval()
    if mutil_gpu:
        model = model.module
    with torch.no_grad():
        samples = model.sample(16, device).cpu().numpy()
        # print(samples)
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28) * 255, cmap='gray')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')

        plt.close(fig)



def train(model, optimizer, lr_scheduler, dataloaders, device, epochs):
    for e in range(epochs):
        for x, y in tqdm(dataloaders['train']):
            model.train()
            x = x.to(device=device)
            y = y.to(device=device)
            gen_image, mu, logvar = model(x)
            loss, BCE, KLD = loss_func(gen_image, x.view(x.shape[0], -1), mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        print('epoche %d, loss = %f, BCE loss = %f, KLD loss = %f' % (e, loss.item(), BCE.item(), KLD.item()))
        logging.info('epoche %d, loss = %f, BCE loss = %f, KLD loss = %f' % (e, loss.item(), BCE.item(), KLD.item()))

        sample(model, device, e)


        
        writer.add_scalars("loss", {"loss":loss.item(), "BCE":BCE.item(), "KLD":KLD.item()}, e)
        save_model(save_dir='model_checkpoint', file_name="check_point", model=model, optimizer = optimizer, lr_scheduler = lr_scheduler)
    save_model(save_dir='model_checkpoint', file_name=task_name,
        model=model, optimizer = optimizer, lr_scheduler = lr_scheduler)
    return model 


def save_model(save_dir, model, optimizer, lr_scheduler, file_name=None):
    if mutil_gpu:
        model = model.module
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if file_name:
        save_path = os.path.join(save_dir, file_name)
    else:
        save_path = os.path.join(save_dir, str(int(time.time())))

    state_dicts = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict()
    }

    torch.save(state_dicts, save_path + '.pkl')

def load_model(file_path, model, optimizer = None, lr_scheduler = None):
    state_dicts = torch.load(file_path, map_location="cpu")
    model.load_state_dict(state_dicts["model"])
    if optimizer:
        optimizer.load_state_dict(state_dicts["optimizer"])
    if lr_scheduler:
        lr_scheduler.load_state_dict(state_dicts["scheduler"])

task_name = "VAE_on_MNIST"
model_name = "VAE"
optimizer_name = 'Adam'
lr = 0.01
weight_decay = 1e-4
step_size = 200
gamma = 0.5
batch_size = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
load_checkpoint = False
mutil_gpu = False
device_ids = ["cuda:0", "cuda:1"]
epochs = 50

logging.basicConfig(filename="{}.log".format(task_name), level=logging.INFO)

logging.info(
    """{}:
    - model name: {}
    - optimizer: {}
    - learning rate: {}
    - weight_decay: {}
    - step_size: {}
    - gamma: {}
    - batch size: {}
    - device : {}
    - epochs: {}
    - load_checkpoint: {}
    - mutil_gpu: {}
    - gpus: {}
 """.format(
        task_name, 
        model_name, 
        optimizer_name, 
        lr, 
        weight_decay,
        step_size,
        gamma,
        batch_size,
        device, 
        epochs,
        load_checkpoint,
        mutil_gpu,
        device_ids)
)
print("""{}:
    - model name: {}
    - optimizer: {}
    - learning rate: {}
    - weight_decay: {}
    - step_size: {}
    - gamma: {}
    - batch size: {}
    - device : {}
    - epochs: {}
    - load_checkpoint: {}
    - mutil_gpu: {}
    - gpus: {}
 """.format(
        task_name, 
        model_name, 
        optimizer_name, 
        lr, 
        weight_decay,
        step_size,
        gamma,
        batch_size,
        device, 
        epochs,
        load_checkpoint,
        mutil_gpu,
        device_ids))

if __name__ == "__main__":
    mean, std = 0, 1
    writer = SummaryWriter()

    model = VAE(input_size = 784, latent_dim = 2)


    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) 
    if load_checkpoint:
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

    train(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloaders=dataLoaders, device=device,
        epochs=epochs)






    
    

