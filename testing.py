import matplotlib.pyplot as plt
from dataset import Aug_GlaS
import numpy as np
from model.model import WESUP
import torch
import torch.nn.functional as F

def cross_entropy(y_hat, y_true, class_weights, epsilon=1e-7):
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

train = Aug_GlaS(transform = True)
a = train.__getitem__(29)
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(a[0].permute(1, 2, 0))
# ax[1].imshow(a[3])
# plt.savefig('testing.png')

model = WESUP(3, 2)
device = torch.device("cuda:2")
opt = torch.optim.Adam(model.parameters(), lr = 1e-4)
model.to(device)
x = a[0].unsqueeze(0).to(device)
y = a[3].to(device)
# print(y.shape)
# print(model)
for i in range(10):
    pred, labels = model(x, y)
    # print(labels)
    labels = convert(labels)
    # print(model)
    print(model.backbone[28].weight)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # print(pred, labels)
    loss = cross_entropy(pred, labels.to(device), class_weights=torch.tensor([0.75, 0.25]).to(device))
    print(loss)
    opt.zero_grad()
    loss.backward()
    pred = pred.detach()
    opt.step()

pred = model(x, y, 'infer')
print(torch.sum(pred == 0), torch.sum(pred == 1))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(pred.cpu().numpy())
ax[1].imshow(y.cpu().numpy())
plt.savefig('testing.png')

#  [[-1.9884e-02, -2.8676e-02, -1.7745e-02],
#           [-1.8820e-02, -2.7692e-02, -3.7976e-02],
#           [-7.1567e-03, -1.6576e-02, -6.9290e-03]],

#          [[-3.3155e-03, -8.4667e-03,  4.0157e-03],
#           [ 1.9905e-02, -1.0356e-02, -4.5904e-04],
#           [ 3.1526e-02,  1.0053e-02,  1.1222e-02]],

#          [[-2.6271e-02, -8.1591e-03, -2.9560e-02],
#           [-3.3923e-02, -2.4079e-02, -2.2005e-02],
#           [-3.4229e-02, -2.6150e-02, -1.4213e-02]]]]