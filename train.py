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
from VAE import loss_fn as loss_func, VAE
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch import Tensor
np.random.seed(0)
torch.manual_seed(0)# 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed_all(0)# 为所有的GPU设置种子，以使得结果是确定的

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def sample(model, device, epoch):
    model.eval()
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
            plt.imshow(sample.reshape(28, 28) * 255, cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')

        plt.close(fig)



def train(model, optimizer, dataloaders, device, epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10e10
    for e in range(epochs):
        for t, (x, y) in enumerate(dataloaders['train']):
            model.train()
            x = x.to(device=device)
            y = y.to(device=device)
            gen_image, mu, logvar = model(x)
            loss, BEC, KLD = loss_func(gen_image, x.view(x.size(0), -1), mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logging.info('epoche %d, loss = %f' % (e, loss.item()))

        sample(model, device, e)

        if loss < best_loss:
            best_loss = loss
            best_model_wts = copy.deepcopy(model.state_dict())
        # rec.append((loss.item(), BEC.item(), KLD.item()))
        save_model(save_dir='model_checkpoint', whole_model=False, file_name="check_point", model=model)
    logging.info('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    save_model(save_dir='model_checkpoint', whole_model=False, file_name=task_name,
        model=model)
    return model 


def save_model(save_dir, whole_model, file_name=None, model=None):
    if mutil_gpu:
        model = model.module
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if file_name:
        save_path = os.path.join(save_dir, file_name)
    else:
        save_path = os.path.join(save_dir, str(int(time.time())))
    if model:
        if whole_model:
            torch.save(model, save_path + '.pkl')
        else:
            torch.save(model.state_dict(), save_path + '.pkl')
    else:
        logging.info('check point not saved, best_model is None')

task_name = "VAE_on_MNIST"
model_name = "VAE"
optimizer_name = 'Adam'
lr = 0.0001
weight_decay = 1e-4
step_size = 50
gamma = 0.5
batch_size = 32
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
load_checkpoint = True
mutil_gpu = True
device_ids = ["cuda:1", "cuda:2"]
epochs = 200

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
    - num_of_classes: {}
    - param_to_update_name_prefix: {}
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
        param_to_update_name_prefix, 
        epochs,
        load_checkpoint,
        mutil_gpu,
        device_ids)
)

if __name__ == "__main__":
    mean, std = 0, 1
    model = VAE()
    if load_checkpoint:
        model.load_state_dict(torch.load("model_checkpoint/VAE_on_MNIST.pkl"))
    if mutil_gpu:
        model = nn.DataParallel(model, device_ids, device)
        model = model.to(device=device)
    params_to_update = []
    for name, param in rnn.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = getattr(optim, optimizer_name)(params_to_update, lr=lr)
    optimizer = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) 

    mnist = torchvision.datasets.MNIST(root = "mnist", train=True, download=True, transform=transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean, std)
            ]
        ))
    dataLoaders = {"train":torch.utils.data.DataLoader(mnist,
            batch_size=batch_size, shuffle=True, num_workers= 0, pin_memory=True, drop_last=False)}

    train(model=model, optimizer=optimizer, dataloaders=dataLoaders, device=device,
        epochs=epochs)






    
    

