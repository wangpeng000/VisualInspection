import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from .utils import makedirpath
from .LRN import LRN
from .datasets import jigsaw_num

__all__ = ['MyJigsawPositionHierEncoder', 'MyJigsawPositionDeepEncoder', 'MyJigsawPositionEncoder', 'MyJigsawPositionClassifier']



def forward_hier(x, emb_small, K):
    K_2 = K // 2
    n = x.size(0)
    x1 = x[..., :K_2, :K_2]
    x2 = x[..., :K_2, K_2:]
    x3 = x[..., K_2:, :K_2]
    x4 = x[..., K_2:, K_2:]
    xx = torch.cat([x1, x2, x3, x4], dim=0)

    hh = emb_small(xx)


    h1 = hh[:n]
    h2 = hh[n: 2 * n]
    h3 = hh[2 * n: 3 * n]
    h4 = hh[3 * n:]


    h12 = torch.cat([h1, h2], dim=3)
    h34 = torch.cat([h3, h4], dim=3)

    h = torch.cat([h12, h34], dim=2)
    return h



xent = nn.CrossEntropyLoss()

class NormalizedLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(NormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # weight的形状
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            # bias的形状
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        with torch.no_grad():
            w = self.weight / self.weight.data.norm(keepdim=True, dim=0)
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class MyJigsawPositionEncoder(nn.Module):
    def __init__(self, K, D=64, bias=True):
        super().__init__()


        self.conv1 = nn.Conv2d(3, 64, 5, 2, 0, bias=bias)
        self.conv2 = nn.Conv2d(64, 64, 5, 2, 0, bias=bias)
        self.conv3 = nn.Conv2d(64, 128, 5, 2, 0, bias=bias)
        self.conv4 = nn.Conv2d(128, D, 5, 1, 0, bias=bias)

        self.K = K
        self.D = D
        self.bias = bias

    def forward(self, x):
        h = self.conv1(x)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv3(h)


        if self.K == 64:
            h = F.leaky_relu(h, 0.1)
            h = self.conv4(h)

        h = torch.tanh(h)

        return h

class MyJigsawPositionDeepEncoder(nn.Module):
    def __init__(self, K, D=64, bias=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=0),
            nn.GroupNorm(12, 96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            LRN(local_size=5, alpha=0.0001, beta=0.75),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            LRN(local_size=5, alpha=0.0001, beta=0.75),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(48, 384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(48, 384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(256, 128, kernel_size=2, stride=1, padding=0),
            nn.GroupNorm(16, 128),

            nn.Conv2d(128, D, kernel_size=2, stride=1, padding=0),
            nn.GroupNorm(int(D/8), D)
        )

        self.K = K
        self.D = D

    def forward(self, x):
        x = self.conv(x)

        x = torch.tanh(x)

        return x

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts/{name}/myjigsawposition_encdeep.pkl'

class MyJigsawPositionHierEncoder(nn.Module):
    def __init__(self, K, D=64, bias=True):
        super().__init__()

        if K > 64:
            self.enc = MyJigsawPositionHierEncoder(K // 2, D, bias=bias)


        elif K == 64:
            self.enc = MyJigsawPositionDeepEncoder(K // 2, D, bias=bias)

        else:
            raise ValueError()

        self.conv1 = nn.Conv2d(D, 128, 2, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(128, D, 1, 1, 0, bias=bias)

        self.K = K
        self.D = D

    def forward(self, x):

        h = forward_hier(x, self.enc, K=self.K)


        h = self.conv1(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = torch.tanh(h)

        return h

    def save(self, name, i):
        fpath = self.fpath_from_name(name, i)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name, i):
        fpath = self.fpath_from_name(name, i)
        self.load_state_dict(torch.load(fpath))
        print('Encoder has been loaded!')

    @staticmethod
    def fpath_from_name(name, i):
        return f'ckpts/{name}/myjigsawposition_enchier_{i}_step.pkl'

class MyJigsawPositionClassifier(nn.Module):
    def __init__(self, K, D, class_num=12):
        super().__init__()
        self.D = D

        self.fc1 = nn.Linear(D, 128)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc2 = nn.Linear(128, 128)
        self.act2 = nn.LeakyReLU(0.1)

        self.fc3 = NormalizedLinear(128, class_num)

        self.K = K

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    def fpath_from_name(self, name):
        return f'ckpts/{name}/position_classifier_K{self.K}.pkl'

    @staticmethod
    def infer(c, enc, batch):

        x1s, x2s, ys = batch
        ys = ys.long().cuda()


        h1 = enc(x1s)
        h2 = enc(x2s)

        logits = c(h1, h2)

        loss = xent(logits, ys)
        return loss

    def forward(self, h1, h2):
        h1 = h1.view(-1, self.D)
        h2 = h2.view(-1, self.D)


        h = h1 - h2

        h = self.fc1(h)
        h = self.act1(h)

        h = self.fc2(h)
        h = self.act2(h)

        h = self.fc3(h)

        return h