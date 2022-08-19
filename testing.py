from audioop import cross
import matplotlib.pyplot as plt
from dataset import GlaS
import numpy as np
from model.model import WESUP
import torch
import torch.nn.functional as F
from utils import box_plot, generate
import pickle

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
testA = GlaS(testA=True)
testB = GlaS(testB = True)
val = torch.utils.data.ConcatDataset([testA, testB])
a = val.__getitem__(59)
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(a[0].permute(1, 2, 0))
# ax[1].imshow(a[1])
# plt.savefig('testing.png')

model = WESUP(3, 2)
device = torch.device("cuda:7")
# opt = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')
# model.load_state_dict(torch.load('Weights/points_10_1.pt'))
# model.to(device)
# x = a[0].to(device)
# y = a[1].to(device)
dice = []
f1 = []

# with open('History/f1_all_20', 'rb') as f:
#     dice = pickle.load(f)

# with open('History/dice_all_20', 'rb') as f:
#     f1 = pickle.load(f)

# print(np.mean(dice), np.mean(f1))
# print(np.std(dice), np.std(f1))

for i in range(1, 6):
    model.load_state_dict(torch.load('Weights/points_10_' + str(i) + '.pt'))
    model.to(device)
    data = box_plot(model, val, device)
    dice.extend(data[1])
    f1.extend(data[0])

dice = np.array(dice)
f1 = np.array(f1)

fig = plt.figure(figsize=(10, 7))

plt.boxplot(dice)
plt.show()
plt.savefig('dice_box_plot_10.png')

plt.close()

plt.boxplot(f1)
plt.show()
plt.savefig('f1_box_plot_10.png')

with open('History/dice_all_10', 'wb') as f:
    pickle.dump(dice, f)

with open('History/f1_all_10', 'wb') as f:
    pickle.dump(f1, f)

print(np.mean(dice), np.mean(f1))
print(np.std(dice), np.std(f1))
# data = box_plot(model, val, device, fn = 'plot.png')
# print(data)
# print(y.shape)
# print(model)
# for i in range(20):
#     l_pred, l_labels, u_pred, u_labels = model(x, y)
#     # print(torch.unique(u_labels))
#     l_labels = convert(l_labels)
#     u_labels = convert(u_labels)
#     # print(model)
#     # print(model.backbone[28].weight)
#     # for name, param in model.named_parameters():
#     #     if param.requires_grad:
#     #         print(name)
#     # print(pred, labels)
#     weights = torch.tensor([3, 1]).to(device)
#     loss = cross_entropy(l_pred, l_labels.to(device), class_weights=weights) + 0.5 * cross_entropy(u_pred, u_labels.to(device), class_weights=weights)
#     print(loss)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     # scheduler.step(loss)
# pred = generate(model, x, y)
# # print(torch.sum(pred == 0), torch.sum(pred == 1))
# fig, ax = plt.subplots(1, 3, figsize=(15, 15))
# ax[0].imshow(a[0].permute(1, 2, 0))
# ax[1].imshow(a[2])
# ax[2].imshow(y.cpu().numpy())
# plt.savefig('testing.png')