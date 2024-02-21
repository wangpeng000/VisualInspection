from codes import mvtecad
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import PatchDataset_NCHW, NHWC2NCHW, distribute_scores

__all__ = ['eval_encoder_NN_multiK', 'eval_embeddings_NN_multiK']


def weights_e_alpha(l2_maps):
    N, I, J, NN = l2_maps.shape
    weights_alpha = np.ones((N, I, J, NN), dtype=np.float32)
    weights_1_alpha = np.ones((N, I, J, NN), dtype=np.float32)
    weights_e_1_alpha = np.ones((N, I, J, NN), dtype=np.float32)

    weights = np.ones((N, I, J, NN), dtype=np.float32)

    l2_maps_e_1_alpha_sum = np.ones((N, I, J, NN), dtype=np.float32)
    result_NN = np.ones((N, I, J, NN), dtype=np.float32)

    l2_maps_sum = np.sum(l2_maps, axis=-1)
    l2_maps_sum[l2_maps_sum == 0] = 1


    for n in range(N):
        for i in range(I):
            for j in range(J):
                weights_alpha[n, i, j, :] = l2_maps[n, i, j, :]/l2_maps_sum[n, i, j]
                weights_alpha[weights_alpha == 0] = 1
                weights_1_alpha[n, i, j, :] = 1 / weights_alpha[n, i, j, :]
                weights_1_alpha[weights_1_alpha > 20] = 15
                weights_e_1_alpha[n, i, j, :] = np.exp(weights_1_alpha[n, i, j, :])
                l2_maps_e_1_alpha_sum[n, i, j] = np.sum(weights_e_1_alpha[n, i, j, :], axis=-1)
                weights[n, i, j, :] = weights_e_1_alpha[n, i, j, :] / l2_maps_e_1_alpha_sum[n, i, j]
                result_NN[n, i, j, :] = l2_maps[n, i, j, :] * weights[n, i, j, :]

    result = np.sum(result_NN, axis=-1)
    return result

def infer(x, enc, K, S):
    x = NHWC2NCHW(x)

    dataset = PatchDataset_NCHW(x, K=K, S=S)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)

    embs = np.empty((dataset.N, dataset.row_num, dataset.col_num, enc.D), dtype=np.float32)  # [-1, I, J, D]

    enc = enc.eval()
    with torch.no_grad():
        for xs, ns, iis, js in loader:
            xs = xs.cuda()

            embedding = enc(xs)
            embedding = embedding.detach().cpu().numpy()

            for embed, n, i, j in zip(embedding, ns, iis, js):
                embs[n, i, j] = np.squeeze(embed)
    return embs


def assess_anomaly_maps(obj, anomaly_maps):
    auroc_seg = mvtecad.segmentation_auroc(obj, anomaly_maps)
    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    auroc_det = mvtecad.detection_auroc(obj, anomaly_scores)
    return auroc_det, auroc_seg


def measure_emb_NN(emb_te, emb_tr, method='kdt', NN=1):
    from .nearest_neighbor import search_NN

    D = emb_tr.shape[-1]

    train_emb_all = emb_tr.reshape(-1, D)

    l2_maps, _ = search_NN(emb_te, train_emb_all, method=method, NN=NN)

    anomaly_maps = weights_e_alpha(l2_maps)

    return anomaly_maps
    
    

def eval_encoder_NN_multiK(enc, obj, maps_num):

    x_tr = mvtecad.get_x_standardized(obj, mode='train')
    x_te = mvtecad.get_x_standardized(obj, mode='test')

    embs64_tr = infer(x_tr, enc, K=64, S=16)
    embs64_te = infer(x_te, enc, K=64, S=16)

    embs32_tr = infer(x_tr, enc.enc, K=32, S=4)
    embs32_te = infer(x_te, enc.enc, K=32, S=4)


    embs64 = embs64_tr, embs64_te
    embs32 = embs32_tr, embs32_te


    return eval_embeddings_NN_multiK(obj, embs64, embs32, NN=maps_num)


def eval_embeddings_NN_multiK(obj, embs64, embs32, NN=1):
    emb_tr, emb_te = embs64

    maps_64 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN)
    maps_64 = distribute_scores(maps_64, (256, 256), K=64, S=16)
    det_64, seg_64 = assess_anomaly_maps(obj, maps_64)

    emb_tr, emb_te = embs32
    maps_32 = measure_emb_NN(emb_te, emb_tr, method='ngt', NN=NN)
    maps_32 = distribute_scores(maps_32, (256, 256), K=32, S=4)
    det_32, seg_32 = assess_anomaly_maps(obj, maps_32)
    
    maps_sum = maps_64 + maps_32
    det_sum, seg_sum = assess_anomaly_maps(obj, maps_sum)
    
    maps_mult = maps_64 * maps_32
    det_mult, seg_mult = assess_anomaly_maps(obj, maps_mult)

    return {
        'det_64': det_64,
        'seg_64': seg_64,

        'det_32': det_32,
        'seg_32': seg_32,

        'det_sum': det_sum,
        'seg_sum': seg_sum,

        'det_mult': det_mult,
        'seg_mult': seg_mult,

        'maps_64': maps_64,
        'maps_32': maps_32,
        'maps_sum': maps_sum,
        'maps_mult': maps_mult,
    }
