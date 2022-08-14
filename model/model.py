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
            nn.Softmax(dim=-1)
        )

    def forward(self, x, y, phase = 'train'):
        if phase == 'train':
            conv = []
            conv.append(F.interpolate(self.backbone.features[:1](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach())
            conv.append(F.interpolate(self.backbone.features[:3](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach())
            conv.append(F.interpolate(self.backbone.features[:6](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach())
            conv.append(F.interpolate(self.backbone.features[:8](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach())
            conv.append(F.interpolate(self.backbone.features[:11](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach())
            conv.append(F.interpolate(self.backbone.features[:13](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach())
            conv.append(F.interpolate(self.backbone.features[:15](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach())
            conv.append(F.interpolate(self.backbone.features[:18](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach())
            conv.append(F.interpolate(self.backbone.features[:20](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach())
            conv.append(F.interpolate(self.backbone.features[:22](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach())
            conv.append(F.interpolate(self.backbone.features[:25](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach())
            conv.append(F.interpolate(self.backbone.features[:27](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach())
            conv.append(F.interpolate(self.backbone.features[:29](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach())
            sp = torch.from_numpy(slic(x.squeeze(0).permute(1, 2, 0).cpu(), n_segments = (x.shape[2] * x.shape[3]) / 400, compactness = 50))
            mN = torch.unique(sp)[-1]
            feat = []
            labels = []
            for i in range(1, mN + 1):
                idxs = (sp == i).nonzero(as_tuple = False)
                r = idxs[:, 0]
                c = idxs[:, 1]

                layers = []
                lb = y[r, c]
                zeros = torch.sum(lb == 0)
                ones = torch.sum(lb == 1)
                twos = torch.sum(lb == 2)
                for layer in conv:
                    s = torch.mean(layer[:, :, r, c], 2).squeeze(0)
                    layers.append(s)

                s = layers[0]
                for i in range(1, len(layers)):
                    s = torch.concat((s, layers[i]))

                feat.append(s.detach())
                lab = torch.argmax(torch.tensor((zeros, ones, twos))).item()
                if lab == 0:
                    if twos > int(len(idxs) / 2):
                        lab = 2
                    elif ones >= 1:
                        lab = 1
                    else:
                        lab = 0
                labels.append(lab)
                
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
                if maX >= 0.8:
                    labels[u[i]] = labels[l[mIdx]]

            pred = []
            for i in feat:
                pred.append(self.classifier(self.mlp(i)))

            return torch.stack(pred), torch.as_tensor(labels)
        elif phase == 'infer':
            with torch.no_grad():
                s = (F.interpolate(self.backbone.features[:1](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach())
                s = torch.concat((s, F.interpolate(self.backbone.features[:3](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach()), dim = 1)
                s = torch.concat((s, F.interpolate(self.backbone.features[:6](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach()), dim = 1)
                s = torch.concat((s, F.interpolate(self.backbone.features[:8](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach()), dim = 1)
                s = torch.concat((s, F.interpolate(self.backbone.features[:11](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach()), dim = 1)
                s = torch.concat((s, F.interpolate(self.backbone.features[:13](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach()), dim = 1)
                s = torch.concat((s, F.interpolate(self.backbone.features[:15](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach()), dim = 1)
                s = torch.concat((s, F.interpolate(self.backbone.features[:18](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach()), dim = 1)
                s = torch.concat((s, F.interpolate(self.backbone.features[:20](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach()), dim = 1)
                s = torch.concat((s, F.interpolate(self.backbone.features[:22](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach()), dim = 1)
                s = torch.concat((s, F.interpolate(self.backbone.features[:25](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach()), dim = 1)
                s = torch.concat((s, F.interpolate(self.backbone.features[:27](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach()), dim = 1)
                s = torch.concat((s, F.interpolate(self.backbone.features[:29](x), size = (x.shape[2], x.shape[2]), mode = 'bicubic').detach()), dim = 1)
                res = torch.zeros(y.shape[0] * y.shape[1])
                s = s.squeeze(0).permute(1, 2, 0).view(-1, 4224)
                val = self.classifier(self.mlp(s))
                res = torch.argmax(val, dim = 1).view(512, 512)
            return res