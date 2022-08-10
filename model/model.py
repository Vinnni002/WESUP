import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from skimage.segmentation import slic
import torch
import numpy as np
from numpy.linalg import norm

def sim(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

class WESUP(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 2, D = 32):
        super(WESUP, self).__init__()
        self.inc = in_channels
        self.out = out_channels
        self.backbone = models.vgg16(pretrained = True)
        self.mlp = nn.Sequential(
            nn.Linear(4224, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, D),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(D, 2),
            nn.Softmax(dim=0)
        )

    def forward(self, x, y):
        conv = []
        conv.append(F.interpolate(self.backbone.features[:1](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic'))
        conv.append(F.interpolate(self.backbone.features[:3](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic'))
        conv.append(F.interpolate(self.backbone.features[:6](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic'))
        conv.append(F.interpolate(self.backbone.features[:8](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic'))
        conv.append(F.interpolate(self.backbone.features[:11](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic'))
        conv.append(F.interpolate(self.backbone.features[:13](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic'))
        conv.append(F.interpolate(self.backbone.features[:15](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic'))
        conv.append(F.interpolate(self.backbone.features[:18](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic'))
        conv.append(F.interpolate(self.backbone.features[:20](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic'))
        conv.append(F.interpolate(self.backbone.features[:22](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic'))
        conv.append(F.interpolate(self.backbone.features[:25](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic'))
        conv.append(F.interpolate(self.backbone.features[:27](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic'))
        conv.append(F.interpolate(self.backbone.features[:29](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic'))
        sp = torch.from_numpy(slic(x.squeeze(0).permute(1, 2, 0).cpu(), n_segments = (x.shape[2] * x.shape[3]) / 500, compactness = 40))
        mN = np.unique(sp)[-1]
        feat = []
        labels = []
        print(torch.unique(y))
        for i in range(1, mN + 1):
            print(i, mN)
            idxs = (sp == i).nonzero(as_tuple = False)
            s1 = 0
            s2 = 0
            s3 = 0
            s4 = 0
            s5 = 0
            s6 = 0
            s7 = 0
            s8 = 0
            s9 = 0
            s10 = 0
            s11 = 0
            s12 = 0
            s13 = 0
            ones = 0
            zeros = 0
            twos = 0
            for idx in idxs:
                s1 += conv[0][:, :, idx[0], idx[1]]
                s2 += conv[1][:, :, idx[0], idx[1]]
                s3 += conv[2][:, :, idx[0], idx[1]]
                s4 += conv[3][:, :, idx[0], idx[1]]
                s5 += conv[4][:, :, idx[0], idx[1]]
                s6 += conv[5][:, :, idx[0], idx[1]]
                s7 += conv[6][:, :, idx[0], idx[1]]
                s8 += conv[7][:, :, idx[0], idx[1]]
                s9 += conv[8][:, :, idx[0], idx[1]]
                s10 += conv[9][:, :, idx[0], idx[1]]
                s11 += conv[10][:, :, idx[0], idx[1]]
                s12 += conv[11][:, :, idx[0], idx[1]]
                s13 += conv[12][:, :, idx[0], idx[1]]
                l = y[idx[0], idx[1]]
                if l == 0:
                    zeros += 1
                elif l == 1:
                    ones += 1
                else:
                    twos += 1

            s1 = s1.squeeze(0) / len(idxs)
            s2 = s2.squeeze(0) / len(idxs)
            s3 = s3.squeeze(0) / len(idxs)
            s4 = s4.squeeze(0) / len(idxs)
            s5 = s5.squeeze(0) / len(idxs)
            s6 = s6.squeeze(0) / len(idxs)
            s7 = s7.squeeze(0) / len(idxs)
            s8 = s8.squeeze(0) / len(idxs)
            s9 = s9.squeeze(0) / len(idxs)
            s10 = s10.squeeze(0) / len(idxs)
            s11 = s11.squeeze(0) / len(idxs)
            s12 = s12.squeeze(0) / len(idxs)
            s13 = s13.squeeze(0) / len(idxs)
            s = torch.concat((s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13))
            feat.append(s)
            lab = np.argmax([zeros, ones, twos])
            if lab == 0:
                if twos > int(len(idxs) / 2):
                    lab = 2
                elif ones >= 1:
                    lab = 1
                else:
                    lab = 0
            labels.append(lab)
        print(labels)
        l = []
        u = []
        for i in range(len(labels)):
            if labels[i] == 0:
                u.append(i)
            else:
                l.append(i)

        affinity = torch.zeros(len(u), len(l))
        for i in range(len(u)):
            maX = 1e-10
            mIdx = 0
            for j in range(len(l)):
                affinity[i][j] = sim(feat[u[i]], feat[l[j]])
                if affinity[i][j] > maX:
                    maX = affinity[i][j]
                    mIdx = j
            if maX > 0.8:
                labels[u[i]] = labels[l[mIdx]]

        pred = []
        for i in feat:
            low_d = self.mlp(i)
            y_hat = self.classifier(low_d)
            print(torch.argmax(y_hat))
            pred.append(torch.argmax(y_hat))

        return pred, labels