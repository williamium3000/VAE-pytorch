import json
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_data_from_tensorboard(path):
    ea=event_accumulator.EventAccumulator(path) 
    ea.Reload()

    val_psnr=ea.scalars.Items('loss')


    data = [i.value for i in val_psnr]
    return data
# CVAE
BCE_loss = load_data_from_tensorboard("runs/Mar17_05-00-30_06bed19cdc6a/loss_BCE/events.out.tfevents.1615957245.06bed19cdc6a.20271.2")
KL_loss = load_data_from_tensorboard("runs/Mar17_05-00-30_06bed19cdc6a/loss_KLD/events.out.tfevents.1615957245.06bed19cdc6a.20271.3")
loss = load_data_from_tensorboard("runs/Mar17_05-00-30_06bed19cdc6a/loss_loss/events.out.tfevents.1615957245.06bed19cdc6a.20271.1")

# VAE
# BCE_loss = load_data_from_tensorboard("runs/Mar17_05-00-23_06bed19cdc6a/loss_BCE/events.out.tfevents.1615957234.06bed19cdc6a.20225.2")
# KL_loss = load_data_from_tensorboard("runs/Mar17_05-00-23_06bed19cdc6a/loss_KLD/events.out.tfevents.1615957234.06bed19cdc6a.20225.3")
# loss = load_data_from_tensorboard("runs/Mar17_05-00-23_06bed19cdc6a/loss_loss/events.out.tfevents.1615957234.06bed19cdc6a.20225.1")


x = list(range(len(KL_loss)))
ax1 = plt.subplot(1,1,1)

ax1.plot(x, BCE_loss, color="red",linewidth=1, label = "BCE loss")
ax1.plot(x, KL_loss, color="blue",linewidth=1, label = "KL loss")
ax1.plot(x, loss, color="yellow",linewidth=1, label = "total loss")

plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss with respect to epoch(CVAE)")
ax1.legend()
plt.show()
