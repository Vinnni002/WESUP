from dataset import GlaS
from model.model import WESUP
import argparse
from torch.utils.data import DataLoader
import tqdm
import torch
import pickle
from utils import convert, cross_entropy, eval
import torch.nn.functional as F

# Parse cmd line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--bs")
parser.add_argument("--ep")
parser.add_argument("--device")
args = parser.parse_args()
devices = args.device.split('-')
devices = [int(i) for i in devices]

# Creating object of dataset class
train = GlaS(transform = True)
testA = GlaS(testA=True)
testB = GlaS(testB=True)
val = torch.utils.data.ConcatDataset([testA, testB])

# Defining DataLoader
bs = int(args.bs)
train_dl = DataLoader(train, shuffle = True, batch_size = bs)
val_dl = DataLoader(val, shuffle = True, batch_size = bs)


# Initialization
device = torch.device('cuda:' + str(devices[0]))
print('Using GPU device :', devices)
model = WESUP(3, 2)
opt = torch.optim.SGD(model.parameters(), lr = 10e-3, momentum = 0.95)
model = torch.nn.DataParallel(model, device_ids = devices)
model = model.to(device)

# Training
epochs = int(args.ep)
max_val = 0
min_val = 10e4
train_history = []
val_history = []
train_metric = []
val_metric = []
    
for epoch in range(epochs):
    total = 0
    total_met = 0
    for (i, (img, img_p, img_anno)) in tqdm.tqdm(enumerate(train_dl), "Epoch : {}".format(epoch + 1)):
        # met = eval(model, val, device)
        img = img.to(torch.float).to(device)
        img_p = img_p.to(torch.float).to(device).squeeze(0)
        pred, labels = model(img, img_p)
        labels = convert(labels)
        loss = cross_entropy(pred, labels.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.detach().cpu()
    
    total = total / (train.__len__() / bs)
    print(total)
    met = eval(model, val, device)
    print(met)
    # total_val = total_val / (val.__len__() / bs)
    # torch.save(model.state_dict(), 'Weights/epoch_' + str(epoch + 1))
    # if ((epoch + 1) % 5 == 0):
    #     met = eval(model, val, device)
    #     print(met)
        # torch.save(model.state_dict(), 'Weights/attn_bce/metric/epoch_' + str(epoch + 1) + '_' + str(met))
    
    # print('Total Loss : {}, F1 Score Training : {}, Total Val Loss : {}, F1 Score Validation : {}'.format(total, total_met, total_val, total_val_met))
    # train_history.append(total)
    # val_history.append(total_val)



with open("history/train_history", "wb") as f:
    pickle.dump(train_history, f)

with open("history/val_history", "wb") as f:
    pickle.dump(val_history, f)

# with open("history/train_metric", "wb") as f:
#     pickle.dump(train_metric, f)

# with open("history/val_metric", "wb") as f:
#     pickle.dump(val_metric, f)