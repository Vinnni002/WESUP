import torch
import glob
import cv2
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
import random
import numpy as np
import torch.nn.functional as F
import math

def find_nearest_white(img, TARGET):
    t1 = (img == 2)
    t2 = (img == 1)
    nonzero = np.argwhere(np.logical_or(t1, t2))
    distances = np.sqrt((nonzero[:,0] - TARGET[0]) ** 2 + (nonzero[:,1] - TARGET[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index]

class Aug_GlaS(Dataset):
    def __init__(self, train = True, testA = False, testB = False, transform = None, pct = 0.9):
        self.transform = transform
        if testA:
            self.name = 'testA'
        elif testB:
            self.name = 'testB'
        else:
            self.name = 'train'
        self.pct = pct

    def __len__(self):
        list = glob.glob('../../Aug_GlaS/' + self.name + '*anno*.bmp')
        return int(len(list))

    def __getitem__(self, idx):
        img = torch.from_numpy(cv2.imread('../../Aug_GlaS/' + self.name + '_' + str(idx + 1) + '.bmp'))
        img_cont = torch.from_numpy(cv2.imread('../../Aug_GlaS/' + self.name + '_' + str(idx + 1) + '_cont.png'))
        img_anno = torch.from_numpy(cv2.imread('../../Aug_GlaS/' + self.name + '_' + str(idx + 1) + '_anno.bmp'))
        if self.name == 'train':
            img_anno = cv2.imread('../../Aug_GlaS/' + self.name + '_' + str(idx + 1) + '_anno.bmp')[:, :, 0]
            nol, totalLabels = cv2.connectedComponents(img_anno.astype('uint8'))
            xs, ys = np.where(totalLabels == 0)
            fg = np.where(totalLabels != 0, totalLabels, 1000)
            num = random.sample(range(1, nol), math.floor(nol * self.pct))
            newImg = np.zeros(totalLabels.shape)

            for i in range(0, nol):
                if i not in num:
                    continue
                tmp = np.where(fg == i, 2, 0)
                newImg = np.add(newImg, tmp) 

            count = 0
            k = 10
            while count <= 10:
                idx = random.sample(range(0, xs.shape[0]), 1)
                a = find_nearest_white(newImg, [xs[idx], ys[idx]])
                if(abs(a[0] - xs[idx]) <= k or abs(a[1] - ys[idx]) <= k):
                    continue    
                count += 1
                print(count)
                x = xs[idx].item()
                y = ys[idx].item()
                newImg[x-k:x+k, y-k : y+k] = 1
            
            img_anno = torch.from_numpy(newImg).unsqueeze(2)

        img = img.permute(2, 0, 1).to(torch.float)
        img_cont = img_cont.permute(2, 0, 1)
        img_anno = img_anno.permute(2, 0, 1)
        
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))

        if self.transform:
            if random.random() > 0.5:
                img = TF.hflip(img)
                img_cont = TF.hflip(img_cont)
                img_anno = TF.hflip(img_anno)

            if random.random() > 0.5:
                img = TF.vflip(img)
                img_cont = TF.vflip(img_cont)
                img_anno = TF.vflip(img_anno)

            angle = random.choice([0, 90, 180, 270])
            img = TF.rotate(img, angle)
            img_cont = TF.rotate(img_cont, angle)
            img_anno = TF.rotate(img_anno, angle)

            r_d = random.uniform(0, 0.1)
            g_d = random.uniform(0, 0.1)
            b_d = random.uniform(0, 0.1)

            img[0, :, :] += r_d
            img[1, :, :] += g_d
            img[2, :, :] += b_d
            img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))

            sig = random.uniform(0, 2)
            gb = transforms.GaussianBlur(kernel_size = 3, sigma = sig)
            img = gb(img)
        
        img_seg = torch.where(img_anno == 0, 0, 1)[0, :, :]
        img_cont = img_cont[0, :, :]
        img_anno = img_anno[0, :, :]

        return img, img_seg, img_cont, img_anno