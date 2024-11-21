import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss
from sklearn.metrics import silhouette_score


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from timm.utils import accuracy
# import utils


def run_kmeans(x, num_cluster, seed=0, args=None):
    """
    Args:
        x: data to be clustered
    """

    # intialize faiss clustering parameters
    d = x.shape[1]
    k = int(num_cluster)
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter = 20
    clus.nredo = 5
    clus.seed = seed
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 10

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = args.gpu
    index = faiss.GpuIndexFlatL2(res, d, cfg)

    clus.train(x, index)

    D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
    im2cluster = [int(n[0]) for n in I]

    # get cluster centroids
    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d) # [k, d]

    # sample-to-centroid distances for each cluster
    Dcluster = [[] for c in range(k)]
    for im, i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])

    sil_score = silhouette_score(x, im2cluster)

    # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(centroids)
    centroids = F.normalize(centroids, p=2, dim=1)

    im2cluster = torch.LongTensor(im2cluster)
    # sil_score = torch.LongTensor([sil_score])


    return centroids.to(args.device), im2cluster.to(args.device), sil_score


def run_kmeans_best_k(x, candidate_K, args):
    """
    Args:
        x: data to be clustered
    """

    # candidate_K = range(2, 4)
    print('performing kmeans clustering')

    results = {'im2cluster': [], 'centroids': [], 'sil_score': []}
    for seed, num_cluster in enumerate(candidate_K):
        centroids, im2cluster, sil_score = run_kmeans(x, num_cluster, seed, args)
        results['im2cluster'].append(im2cluster)
        results['centroids'].append(centroids)
        results['sil_score'].append(sil_score)


    highest_sil = max(results['sil_score'])
    best_ind = results['sil_score'].index(highest_sil)


    # # convert to cuda Tensors for broadcast
    centroids_best = results['centroids'][best_ind]
    im2cluster_best = results['im2cluster'][best_ind]
    best_K = candidate_K[best_ind]
    print("best K:", best_K)

    return {'im2cluster': im2cluster_best, 'centroids': centroids_best, 'best_K': best_K}



@torch.no_grad()
def get_features_ema(args, eval_loader, model, model_ema, data_type="train"):
    model.eval()
    model = model.cuda()

    classname_emb = model.base_text_features
    dtype = model.dtype

    targets, pseudo_targets, logits, logits_emas = [], [], [], []
    # features_stu = torch.zeros(len(eval_loader.dataset), args.low_dim).cuda()
    features_ema = torch.zeros(len(eval_loader.dataset), args.low_dim).cuda()

    for i, (((images_weak, images_strong, _), target_), image_id) in enumerate(eval_loader):
        images_weak = images_weak.cuda()
        # images_strong= images_strong.cuda()

        # # student
        # logits_, feature_, _, _ = model(images_strong.type(dtype))
        # features_stu[image_id] = feature_
        #
        # logits_ = F.softmax(logits_, dim=-1)
        # logits.append(logits_)

        # teacher
        feature_ema_batch, _ = model_ema.ema.encode_image(images_weak.type(dtype))
        features_ema[image_id] = feature_ema_batch


        # logits_ema_ = F.softmax(logits_ema_, dim=-1)
        # logits_emas.append(logits_ema_)
        #
        # max_text_prob, pseudo_target_ = torch.max(logits_ema_, dim=1)
        # pseudo_targets.append(pseudo_target_)

        targets.append(target_)


    targets = torch.cat(targets).int()
    # pseudo_targets = torch.cat(pseudo_targets)
    # logits_emas = torch.cat(logits_emas)

    dist.barrier()
    # dist.all_reduce(features_stu, op=dist.ReduceOp.SUM)
    dist.all_reduce(features_ema, op=dist.ReduceOp.SUM)

    # return features_stu, features_ema, targets, pseudo_targets, logits_emas
    return features_ema, targets, classname_emb


def get_features_ema_per_label(args, eval_loader, model, model_ema, data_type = "train"):
    model.eval()
    model = model.cuda()

    classname_emb = model.base_text_features
    dtype = model.dtype

    targets, pseudo_targets, logits, logits_emas = [], [], [], []
    # features_stu = torch.zeros(len(eval_loader.dataset), args.low_dim).cuda()
    features_ema = torch.zeros(len(eval_loader.dataset), args.low_dim).cuda()

    for i, (((images_weak, images_strong, _), target_), image_id) in enumerate(eval_loader):
        with torch.no_grad():
            images_weak = images_weak.cuda()
            # images_strong= images_strong.cuda()

            # # student
            # logits_, feature_, _, _ = model(images_strong.type(dtype))
            # features_stu[image_id] = feature_
            #
            # logits_ = F.softmax(logits_, dim=-1)
            # logits.append(logits_)

            # teacher
            feature_ema_batch, _ = model_ema.ema.encode_image(images_weak.type(dtype))
            features_ema[image_id] = feature_ema_batch


            # logits_ema_ = F.softmax(logits_ema_, dim=-1)
            # logits_emas.append(logits_ema_)
            #
            # max_text_prob, pseudo_target_ = torch.max(logits_ema_, dim=1)
            # pseudo_targets.append(pseudo_target_)

            targets.append(target_)


    targets = torch.cat(targets).int()
    # pseudo_targets = torch.cat(pseudo_targets)
    # logits_emas = torch.cat(logits_emas)

    dist.barrier()
    # dist.all_reduce(features_stu, op=dist.ReduceOp.SUM)
    dist.all_reduce(features_ema, op=dist.ReduceOp.SUM)

    # return features_stu, features_ema, targets, pseudo_targets, logits_emas
    return features_ema, targets, classname_emb



