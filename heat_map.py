import argparse
import matplotlib.pyplot as plt
from codes import mvtecad
from tqdm import tqdm
from codes.utils import resize, makedirpath

from skimage import morphology
from skimage.segmentation import mark_boundaries
import os
import numpy as np
import matplotlib
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve

parser = argparse.ArgumentParser()
parser.add_argument('--obj', default='transistor')
args = parser.parse_args()


def plot_fig(test_img, scores, gts, threshold, obj):

    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        gt = gts[i]
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(20, 5))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[1].imshow(gt, cmap='gray')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[4].imshow(vis_img)
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }

        fpath = f'anomaly_maps/{obj}/{i:03d}.png'
        makedirpath(fpath)
        fig_img.savefig(fpath)
        plt.close()

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x * std) + mean) * 255.).astype(np.uint8)

    return x


def main():
    from codes.inspection import eval_encoder_NN_multiK
    from codes.networks import MyJigsawPositionHierEncoder

    obj = args.obj

    enc = MyJigsawPositionHierEncoder(K=64, D=64).cuda()
    enc.load(obj, 0)
    enc.eval()
    results = eval_encoder_NN_multiK(enc, obj, 1)
    score_map = results['maps_mult']

    images = mvtecad.get_x(obj, mode='test')

    masks = mvtecad.get_mask(obj)
    masks[masks==255] = 1


    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=2)

    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    gt_mask = np.asarray(masks)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    plot_fig(images, scores, masks, threshold, obj)


if __name__ == '__main__':
    main()
