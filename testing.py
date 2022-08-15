import matplotlib.pyplot as plt
from dataset import GlaS
import numpy as np
from model.model import WESUP
import torch
import torch.nn.functional as F

def cross_entropy(y_hat, y_true, class_weights = None, epsilon=1e-7):
    """Semi-supervised cross entropy loss function.

    Args:
        y_hat: prediction tensor with size (N, C), where C is the number of classes
        y_true: label tensor with size (N, C). A sample won't be counted into loss
            if its label is all zeros.
        class_weights: class weights tensor with size (C,)
        epsilon: numerical stability term

    Returns:
        cross_entropy: cross entropy loss computed only on samples with labels
    """

    device = y_hat.device

    # clamp all elements to prevent numerical overflow/underflow
    y_hat = torch.clamp(y_hat, min=epsilon, max=(1 - epsilon))

    # number of samples with labels
    labeled_samples = torch.sum(y_true.sum(dim=1) > 0).float()

    if labeled_samples.item() == 0:
        return torch.tensor(0.).to(device)

    ce = -y_true * torch.log(y_hat)

    if class_weights is not None:
        ce = ce * class_weights.unsqueeze(0).float()

    return torch.sum(ce) / labeled_samples

def convert(labels):
    l = torch.zeros(labels.shape[0], 2)
    idx = (labels == 1).nonzero(as_tuple = False)
    l[idx, 0] = 1
    idx = (labels == 2).nonzero(as_tuple = False)
    l[idx, 1] = 1
    return l

train = GlaS(transform = True)
a = train.__getitem__(20)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(a[0].permute(1, 2, 0))
ax[1].imshow(a[1])
plt.savefig('testing.png')

model = WESUP(3, 2)
device = torch.device("cuda:2")
opt = torch.optim.Adam(model.parameters(), lr = 1e-4)
# model.load_state_dict(torch.load('Weights/epoch_1'))
model.to(device)
x = a[0].unsqueeze(0).to(device)
y = a[1].to(device)
# print(y.shape)
# print(model)
for i in range(20):
    pred, labels = model(x, y)
    # print(labels)
    labels = convert(labels)
    # print(model)
    # print(model.backbone[28].weight)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # print(pred, labels)
    loss = cross_entropy(pred, labels.to(device))
    print(loss)
    opt.zero_grad()
    loss.backward()
    pred = pred.detach()
    opt.step()

pred = model(x, y, 'infer')
print(torch.sum(pred == 0), torch.sum(pred == 1))
fig, ax = plt.subplots(1, 4)
ax[0].imshow(pred.cpu().numpy())
ax[1].imshow(y.cpu().numpy())
ax[2].imshow(a[0].permute(1, 2, 0))
ax[3].imshow(a[2])
plt.savefig('testing.png')