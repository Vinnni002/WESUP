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
opt = torch.optim.Adam(model.parameters(), lr = 10e-4, weight_decay = 10e-5)
# model.load_state_dict(torch.load('Weights/ep.pt', map_location = device))
# opt = torch.optim.SGD(model.parameters(), lr = 10e-4, momentum = 0.95)
criterion = torch.nn.BCEWithLogitsLoss()
model = model.to(device)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')

# Training
epochs = int(args.ep)
max_val = 0
min_val = 10e4
train_history = []
val_history = []
train_metric = []
val_metric = []
history = []
for epoch in range(epochs):
    total = 0
    total_met = 0
    for (i, (img, img_p, img_anno)) in tqdm.tqdm(enumerate(train_dl), "Epoch : {}".format(epoch + 1)):
        img = img.to(torch.float).to(device)
        img_p = img_p.to(torch.float).to(device).squeeze(0)
        l_pred, l_labels, u_pred, u_labels = model(img, img_p)
        # print(l_labels.shape, l_pred.shape)
        # print(l_labels)
        l_labels, l_pred = convert(l_labels, l_pred)
        # print(l_labels)
        # print(l_labels.shape, l_pred.shape)
        u_labels, u_pred = convert(u_labels, u_pred)
        # loss = cross_entropy(l_pred, l_labels.to(device))
        loss = criterion(l_pred.float(), l_labels.to(device).float()) + 0.5 * criterion(u_pred.float(), u_labels.to(device).float())
        # loss = cross_entropy(l_pred, l_labels.to(device)) + 0.5 * cross_entropy(u_pred, u_labels.to(device))
        opt.zero_grad()
        print(loss)
        loss.backward()
        opt.step()
        total += loss.detach().cpu()
    
    total = total / (train.__len__() / bs)
    print(total)
    met = eval(model, val, device)
    print(met)
    history.append(met)
    with open('History/history_6', 'wb') as f:
        pickle.dump(history, f)

    if met[0] > max_val:
        max_val = met[0]
        torch.save(model.state_dict(), 'Weights/best_6.pt')