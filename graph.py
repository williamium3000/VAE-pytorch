import json
import numpy as np
import matplotlib.pyplot as plt

data = json.load(open("VAE_on_MNIST.json"))
BCE_loss = []
KL_loss = []
for piece in data:
    BCE_loss.append(piece[1])
    KL_loss.append(piece[2])

x = list(range(len(BCE_loss)))




ax1 = plt.subplot(1,1,1)

ax1.plot(x, BCE_loss, color="red",linewidth=1, label = "BCE loss")
# ax1.plot(x, KL_loss, color="blue",linewidth=1, label = "KL loss")

plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("KL loss with respect to epoch")
ax1.legend()
# plt.show()
plt.savefig("BCE-loss.png")