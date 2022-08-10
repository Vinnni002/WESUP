import matplotlib.pyplot as plt
from dataset import Aug_GlaS
import numpy as np
from model.model import WESUP
import torch

train = Aug_GlaS(transform = True)
a = train.__getitem__(90)

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(a[0].permute(1, 2, 0))
# ax[1].imshow(a[3])
# plt.savefig('testing.png')

model = WESUP(3, 2)
device = torch.device("cuda:2")
model.to(device)
x = model(a[0].unsqueeze(0).to(device), a[3].to(device))