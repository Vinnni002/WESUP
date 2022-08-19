import torch
import glob
import cv2
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
import random
import numpy as np
import torch.nn.functional as F

class GlaS(Dataset):
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
        list = glob.glob('data_glas/GlaS/' + self.name + '*anno*.bmp')
        return int(len(list))

    def __getitem__(self, idx):
        siz = 400
        img = torch.from_numpy(cv2.resize(cv2.imread('data_glas/GlaS/' + self.name + '_' + str(idx + 1) + '.bmp'), (siz, siz), cv2.INTER_CUBIC))
        img_point = torch.from_numpy(cv2.resize(cv2.imread('data_glas/points_10_5/' + self.name + '_' + str(idx + 1) + '_anno.bmp'), (siz, siz), cv2.INTER_CUBIC))
        img_anno = torch.from_numpy(cv2.resize(cv2.imread('data_glas/GlaS/' + self.name + '_' + str(idx + 1) + '_anno.bmp'), (siz, siz), cv2.INTER_CUBIC))

        img = img.permute(2, 0, 1).to(torch.float)
        img_point = img_point.permute(2, 0, 1)
        img_anno = img_anno.permute(2, 0, 1)
        
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))

        if self.transform:
            if random.random() > 0.5:
                img = TF.hflip(img)
                img_point = TF.hflip(img_point)
                img_anno = TF.hflip(img_anno)

            if random.random() > 0.5:
                img = TF.vflip(img)
                img_point = TF.vflip(img_point)
                img_anno = TF.vflip(img_anno)

            angle = random.choice([0, 90, 180, 270])
            img = TF.rotate(img, angle)
            img_point = TF.rotate(img_point, angle)
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
        
        img_point = img_point[0, :, :]
        img_anno = img_anno[0, :, :]
    
        return img, img_point, img_anno