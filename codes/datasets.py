from .utils import *
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

__all__ = ['SVDD_Dataset', 'PositionDataset', 'MyJigsawPositionDataset']

jigsaw_num = 4

def generate_coords(H, W, K):
    h = np.random.randint(0, H - K + 1)
    w = np.random.randint(0, W - K + 1)
    return h, w


def generate_coords_mine(H, W, K):
    h = np.random.randint(K, H - K + 1)
    w = np.random.randint(K, W - K + 1)
    return h, w

def generate_coords_position(H, W, K):
    with task('P1'):
        p1 = generate_coords(H, W, K)
        h1, w1 = p1

    pos = np.random.randint(8)

    with task('P2'):
        J = K // 4

        K3_4 = 3 * K // 4
        h_dir, w_dir = pos_to_diff[pos]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h2 = h1 + h_diff
        w2 = w1 + w_diff

        h2 = np.clip(h2, 0, H - K)
        w2 = np.clip(w2, 0, W - K)

        p2 = (h2, w2)

    return p1, p2, pos


def generate_coords_svdd(H, W, K):

    with task('P1'):
        p1 = generate_coords(H, W, K)
        h1, w1 = p1

    with task('P2'):
        J = K // 32

        h_jit, w_jit = 0, 0

        while h_jit == 0 and w_jit == 0:
            h_jit = np.random.randint(-J, J + 1)
            w_jit = np.random.randint(-J, J + 1)

        h2 = h1 + h_jit
        w2 = w1 + w_jit

        h2 = np.clip(h2, 0, H - K)
        w2 = np.clip(w2, 0, W - K)

        p2 = (h2, w2)

    return p1, p2


pos_to_diff = {
    0: (-1, -1),
    1: (-1, 0),
    2: (-1, 1),
    3: (0, -1),
    4: (0, 1),
    5: (1, -1),
    6: (1, 0),
    7: (1, 1)
}


def generate_coords_position_mine(H, W, K):
    with task('P_STD'):
        p_std = generate_coords(H, W, K)
        h_std, w_std = p_std

    with task('P_STD2'):
        J = K // 32
        h_jit, w_jit = 0, 0

        while h_jit == 0 and w_jit == 0:
            h_jit = np.random.randint(-J, J + 1)
            w_jit = np.random.randint(-J, J + 1)

        h_std2 = h_std + h_jit
        w_std2 = h_std + h_jit

        h_std2 = np.clip(h_std2, 0, H - K)
        w_std2 = np.clip(w_std2, 0, H - K)

        p_std2 = (h_std2, w_std2)

    with task('P0'):
        pos0 = 0

        J = K // 4
        K3_4 = 3 * K //4
        h_dir, w_dir = pos_to_diff[pos0]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h0 = h_std + h_diff
        w0 = w_std + w_diff

        h0 = np.clip(h0, 0, H - K)
        w0 = np.clip(w0, 0, W - K)

        p0 = (h0, w0)

    with task('P1'):
        pos1 = 1

        J = K // 4
        K3_4 = 3 * K // 4
        h_dir, w_dir = pos_to_diff[pos1]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h1 = h_std + h_diff
        w1 = w_std + w_diff

        h1 = np.clip(h1, 0, H - K)
        w1 = np.clip(w1, 0, W - K)

        p1 = (h1, w1)

    with task('P2'):
        pos2 = 2

        J = K // 4
        K3_4 = 3 * K // 4
        h_dir, w_dir = pos_to_diff[pos2]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h2 = h_std + h_diff
        w2 = w_std + w_diff

        h2 = np.clip(h2, 0, H - K)
        w2 = np.clip(w2, 0, W - K)

        p2 = (h2, w2)

    with task('P3'):
        pos3 = 3

        J = K // 4
        K3_4 = 3 * K // 4
        h_dir, w_dir = pos_to_diff[pos3]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h3 = h_std + h_diff
        w3 = w_std + w_diff

        h3 = np.clip(h3, 0, H - K)
        w3 = np.clip(w3, 0, W - K)

        p3 = (h3, w3)

    with task('P4'):
        pos4 = 4

        J = K // 4
        K3_4 = 3 * K // 4
        h_dir, w_dir = pos_to_diff[pos4]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h4 = h_std + h_diff
        w4 = w_std + w_diff

        h4 = np.clip(h4, 0, H - K)
        w4 = np.clip(w4, 0, W - K)

        p4 = (h4, w4)

    with task('P5'):
        pos5 = 5

        J = K // 4
        K3_4 = 3 * K // 4
        h_dir, w_dir = pos_to_diff[pos5]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h5 = h_std + h_diff
        w5 = w_std + w_diff

        h5 = np.clip(h5, 0, H - K)
        w5 = np.clip(w5, 0, W - K)

        p5 = (h5, w5)

    with task('P6'):
        pos6 = 6

        J = K // 4
        K3_4 = 3 * K // 4
        h_dir, w_dir = pos_to_diff[pos6]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h6 = h_std + h_diff
        w6 = w_std + w_diff

        h6 = np.clip(h6, 0, H - K)
        w6 = np.clip(w6, 0, W - K)

        p6 = (h6, w6)

    with task('P7'):
        pos7 = 7

        J = K // 4
        K3_4 = 3 * K // 4
        h_dir, w_dir = pos_to_diff[pos7]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h7 = h_std + h_diff
        w7 = w_std + w_diff

        h7 = np.clip(h7, 0, H - K)
        w7 = np.clip(w7, 0, W - K)

        p7 = (h7, w7)

    return p0, p1, p2, p3, p_std2,p4, p5, p6, p7, p_std




class SVDD_Dataset(Dataset):

    def __init__(self, memmap, K=64, repeat=1):
        super().__init__()
        self.arr = np.asarray(memmap)
        self.K = K
        self.repeat = repeat

    def __len__(self):
        N = self.arr.shape[0]
        return N * self.repeat

    def __getitem__(self, idx):
        N = self.arr.shape[0]
        K = self.K
        n = idx % N

        p1, p2 = generate_coords_svdd(256, 256, K)

        image = self.arr[n]

        patch1 = crop_image_CHW(image, p1, K)
        patch2 = crop_image_CHW(image, p2, K)

        return patch1, patch2

    @staticmethod
    def infer(enc, batch):

        x1s, x2s, = batch
        h1s = enc(x1s)
        h2s = enc(x2s)
        diff = h1s - h2s
        l2 = diff.norm(dim=1)
        loss = l2.mean()

        return loss


class PositionDataset(Dataset):

    def __init__(self, x, K=64, repeat=1):
        super(PositionDataset, self).__init__()
        self.x = np.asarray(x)
        self.K = K
        self.repeat = repeat

    def __len__(self):
        N = self.x.shape[0]
        return N * self.repeat

    def __getitem__(self, idx):
        N = self.x.shape[0]
        K = self.K
        n = idx % N

        image = self.x[n]
        p1, p2, pos = generate_coords_position(256, 256, K)

        patch1 = crop_image_CHW(image, p1, K).copy()
        patch2 = crop_image_CHW(image, p2, K).copy()

        rgbshift1 = np.random.normal(scale=0.02, size=(3, 1, 1))
        rgbshift2 = np.random.normal(scale=0.02, size=(3, 1, 1))

        patch1 += rgbshift1
        patch2 += rgbshift2


        noise1 = np.random.normal(scale=0.02, size=(3, K, K))
        noise2 = np.random.normal(scale=0.02, size=(3, K, K))

        patch1 += noise1
        patch2 += noise2

        return patch1, patch2, pos

class MyJigsawPositionDataset(Dataset):
    def __init__(self, x, K=64, repeat=1):
        super(MyJigsawPositionDataset, self).__init__()
        self.x = np.asarray(x)
        self.K = K
        self.repeat = repeat

    def __len__(self):
        N = self.x.shape[0]
        return N * self.repeat

    def __getitem__(self, idx):
        N = self.x.shape[0]
        K = self.K
        n = idx % N

        image = self.x[n]

        position = generate_coords_position_mine(256, 256, K)
        npy = np.load('data.npy')
        order = np.random.randint(len(npy))

        patch1 = crop_image_CHW(image, position[npy[order][0]], K).copy()
        patch2 = crop_image_CHW(image, position[npy[order][1]], K).copy()
        pos = npy[order][2]

        rgbshift1 = np.random.normal(scale=0.02, size=(3, 1, 1))
        rgbshift2 = np.random.normal(scale=0.02, size=(3, 1, 1))

        patch1 += rgbshift1
        patch2 += rgbshift2


        noise1 = np.random.normal(scale=0.02, size=(3, K, K))
        noise2 = np.random.normal(scale=0.02, size=(3, K, K))

        patch1 += noise1
        patch2 += noise2

        return patch1, patch2, pos