import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from skimage.segmentation import slic
import torch
import numpy as np
from numpy.linalg import norm
import math
import matplotlib.pyplot as plt

def sim(a, b):
    return math.exp(-1 * (torch.dot(a, b) / (torch.norm(a) * torch.norm(b))))

class WESUP(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 2, D = 32):
        super(WESUP, self).__init__()
        self.inc = in_channels
        self.out = out_channels
        self.backbone = models.vgg16(pretrained = True).features
        
        # sum of channels of all feature maps
        self.fm_channels_sum = 0

        # side convolution layers after each conv feature map
        for layer in self.backbone:
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self._hook_fn)
                setattr(self, f'side_conv{self.fm_channels_sum}',
                        nn.Conv2d(layer.out_channels, layer.out_channels // 2, 1))
                self.fm_channels_sum += layer.out_channels // 2

        self.mlp = nn.Sequential(
            nn.Linear(2112, 1024),
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

    def _hook_fn(self, _, input_, output):
        if self.feature_maps is None:
            self.fm_size = (input_[0].size(2), input_[0].size(3))
            side_conv_name = 'side_conv0'
        else:
            side_conv_name = f'side_conv{self.feature_maps.size(0)}'

        output = getattr(self, side_conv_name)(output.clone())
        output = F.interpolate(output, self.fm_size,
                               mode='bilinear', align_corners=True)

        if self.feature_maps is None:
            self.feature_maps = output.squeeze()
        else:
            self.feature_maps = torch.cat(
                (self.feature_maps, output.squeeze()))

    def forward(self, x, y, phase = 'train'):
        if phase == 'train':
            self.feature_maps = None
            _ = self.backbone(x)
            sp = torch.from_numpy(slic(x.squeeze(0).permute(1, 2, 0).cpu(), n_segments = (x.shape[2] * x.shape[3]) / 150, compactness = 40))
            mN = torch.unique(sp)[-1]
            # print(mN)
            feat = []
            labels = []
            pred = []
            # print(torch.unique(sp))
            for i in range(1, mN + 1):
                idxs = (sp == i).nonzero(as_tuple = False)
                # tmp = (sp == i)
                # plt.imshow(tmp)
                # plt.savefig('t.png')
                r = idxs[:, 0]
                c = idxs[:, 1]

                tmp = torch.mean(self.feature_maps[:, r, c], 1)
                pred.append(self.classifier(self.mlp(tmp)))
                feat.append(tmp.detach())
                
                lb = y[r, c]
                zeros = torch.sum(lb == 0)
                ones = torch.sum(lb == 1)
                twos = torch.sum(lb == 2)
                
                lab = torch.argmax(torch.tensor((zeros, ones, twos))).item()
                if lab == 0:
                    if twos > ones:
                        lab = 2
                    elif ones > twos:
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
            count = 0
            affinity = torch.zeros(len(u), len(l))
            for i in range(len(u)):
                maX = 1e-10
                mIdx = 0
                for j in range(len(l)):
                    affinity[i][j] = sim(feat[u[i]], feat[l[j]])
                    if affinity[i][j] > maX:
                        maX = affinity[i][j]
                        mIdx = j
                if maX >= 0.7:
                    count += 1
                    labels[u[i]] = labels[l[mIdx]]
                # print(maX)
            # pred = []
            # for i in feat:
            #     pred.append(self.classifier(self.mlp(i)))
            print(count)
            return torch.stack(pred), torch.as_tensor(labels)
        elif phase == 'infer':
            with torch.no_grad():
                self.feature_maps = None
                _ = self.backbone(x)
                res = torch.zeros(y.shape[0] * y.shape[1])
                self.feature_maps = self.feature_maps.squeeze(0).permute(1, 2, 0).view(-1, 2112)
                val = self.classifier(self.mlp(self.feature_maps))
                res = torch.argmax(val, dim = 1).view(256, 256)
            return res