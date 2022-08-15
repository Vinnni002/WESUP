import matplotlib.pyplot as plt
from dataset import GlaS
import numpy as np
from model.model import WESUP
import torch
import torch.nn.functional as F
import time

train = GlaS(transform = True)

k = train.__len__()
print(k)

for i in range(k):
    print(i)
    a = train.__getitem__(i)
    fig, ax = plt.subplots(1, 3)
    ax[0].set_title(i)
    ax[0].imshow(a[0].permute(1, 2, 0))
    ax[1].imshow(a[1])
    ax[2].imshow(a[2])
    plt.savefig('t.png')
    time.sleep(2)