import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import json
import os
from contextlib import suppress
import random
import dgl
import pdb
from my_utils.utils import cal_gap


from pathlib import Path

from ema import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from my_utils import utils
from my_utils.utils import NativeScalerWithGradNormCount as NativeScaler

from build_dataset import build_dataset
from engine_self_training import evaluate, train_one_epoch

from model import clip_classifier, CHGCLIP, GCACLIP
from my_utils.utils import mkdir_if_missing
from my_utils.utils import Logger
from my_utils.kmeans_proto import get_features_ema, run_kmeans_best_k,run_kmeans
import torch.distributed as dist


import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser('UHS_GAT training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)

    
    # CLIP parameters
    parser.add_argument("--template", default='templates.json', type=str)
    parser.add_argument("--classname", default='classes.json', type=str)
    parser.add_argument('--clip_model', default='ViT-B/16', help='pretrained clip model name') 
    parser.add_argument('--image_mean', default=(0.48145466, 0.4578275, 0.40821073)) 
    parser.add_argument('--image_std', default=(0.26862954, 0.26130258, 0.27577711)) 
    parser.add_argument('--input_size', default=224, type=int, help='images input size') 
  
    # training parameters
    parser.add_argument("--train_config", default='train_configs.json', type=str, help='training configurations') 
    parser.add_argument('--mask', action='store_false')
    # parser.set_defaults(mask=True)
    parser.set_defaults(mask=False)
    parser.add_argument('--model_ema_decay', type=float
                        , default=0.9998, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    parser.add_argument("--wreg", type=float, default=0.0, help="trade-off weight for regularization loss")

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.65)
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--center_warmup_epochs', type=int, default=0,
                        help='number of warm-up epochs to only train with InfoNCE loss')

    # Augmentation parameters  
    parser.add_argument('--train_crop_min', default=0.3, type=float)
    parser.add_argument('--color_jitter', type=float, default=0, metavar='PCT')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int, help='number of the classification types')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')
    
    parser.add_argument('--output_dir', default='', help='path to save checkpoint and log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=False)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--low_dim', default=512, type=int,
                        help='feature dimension (default: 512)')
    parser.add_argument('--num_cluster', type=str, default=1, help='Multiples of the number of categories')
    parser.add_argument('--m_ncls', type=int, default=2, help='Multiples of the number of categories')

    parser.add_argument('--l1', default='1', type=float, help='Masked Image Modeling loss')
    parser.add_argument('--l2', default='1', type=float, help='Patch Alignment loss')
    parser.add_argument('--l3', default='1', type=float, help='Hierarchical Cross-modal Alignment loss')

    parser.add_argument('--alpha', default='0.5', type=float, help='weight of feature fuse')

    # parameters for prototypes
    parser.add_argument('--fine_grained_count', default='10', type=int, help='fine-grained count')
    parser.add_argument('--label_set_size', default=10, type=int, help='subclass size')
    parser.add_argument("--granularity", default='CF', type=str, choices=['C', 'F', 'CF'], help="C for Coarse-only prototypes, F for Fine-only, CF for fine+only")

    return parser.parse_args()


def main(args):

    utils.init_distributed_mode(args)


    # config
    train_configs = json.load(open(args.train_config,'r'))
    train_config = train_configs[args.dataset+'_'+args.clip_model]

    now = datetime.datetime.now()
    args.version2 = args.dataset + '_' + now.strftime('%m.%d_%H:%M')


    # log
    import sys
    # log_dir = '/public/gsp/code/UHS_GAT/'
    logfile_dir = os.path.join(os.getcwd(), f'logs/{args.granularity}_reg{args.wreg}/')

    mkdir_if_missing(logfile_dir)
    nCluster = 2
    # logfile_name = logfile_dir + args.version2 + ".log"
    logfile_name = logfile_dir + args.dataset + '_' + now.strftime('%m.%d_%H:%M') + ".log"
    print("log_file_path", logfile_name)
    sys.stdout = Logger(filename=logfile_name, stream=sys.stdout)

    # save training stage
    if not args.output_dir:
        # args.output_dir = os.path.join('output',args.dataset+"/"+ loss_efficiency_exp)
        args.output_dir = os.path.join('output', args.dataset, args.granularity)
        if args.mask:
            args.output_dir = os.path.join(args.output_dir, "%s_mpatch%d_mratio%.1f_walign%.1f_tau%.1f_epoch%d_lr%.5f_alpha.1f"%(args.clip_model[:5],train_config['mask_patch_size'],train_config['mask_ratio'],train_config['w_align'],train_config['conf_threshold'],train_config['epochs'], train_config['lr']))
        else:
            args.output_dir = os.path.join(args.output_dir, "%s_epoch%d_lr%.5f_bs%d_wreg%.3f" % (
        args.clip_model.replace("/", ""), train_config['epochs'], train_config['lr'], args.batch_size, args.wreg))
    else:
        # args.output_dir = os.path.join(args.output_dir, args.dataset+"/"+ loss_efficiency_exp)
        args.output_dir = os.path.join('output', args.dataset, args.granularity)
        args.output_dir = os.path.join(args.output_dir, "%s_epoch%d_lr%.5f_bs%d_wreg%.3f" % (
        args.clip_model.replace("/", ""), train_config['epochs'], train_config['lr'], args.batch_size, args.wreg))


    args.output_dir += now.strftime('%m.%d_%H:%M')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    
    # create CHGCLIP from pretrained clip model
    # model = CHGCLIP(args) # GAT
    model = GCACLIP(args)
    model = model.to(args.device)


    # model = clip_classifier(args)
    # 多次聚类，改成一次聚类
    args.nb_classes = len(model.classnames)
    args.num_cluster = [2,4,6,8,10]
    # args.num_cluster.append(args.nb_classes * args.m_ncls)
    # args.num_cluster.append(nCluster)
    # if args.num_cluster:
    #     for fc in range(3, args.fine_grained_count + 1, 2):
    #         args.num_cluster.append(args.nb_classes * (fc + 1))
    #         break
    # else:
    #     for fc in range(1, args.fine_grained_count + 1, 2):
    #         args.num_cluster.append(args.nb_classes * (fc + 1))
    #         break

    # dataset
    dataset_train = build_dataset(is_train=True, args=args, train_config=train_config)
    dataset_val = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=False)

    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.output_dir)
    else:
        log_writer = None
    if args.output_dir and utils.is_main_process():    
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(dict(args._get_kwargs())) + "\n")


    # dataloader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=2*args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    # for ema_model while training
    eval_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=eval_sampler,
        batch_size=args.batch_size*5,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    # model
    model_ema = ModelEma(
        model,
        decay=args.model_ema_decay,
        resume='')
    print("Using EMA with decay = %.5f" % (args.model_ema_decay) )


    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_batch_size = args.batch_size * utils.get_world_size()
    num_training_steps_per_epoch = len(data_loader_train)

    args.alpha = train_config['alpha']
    args.lr = train_config['lr'] * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.epochs = train_config['epochs']
    args.eval_freq = train_config['eval_freq']
    print("alpha = %.1f" % args.alpha)
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training examples = %d" % len(dataset_train))

    num_layers = model_without_ddp.model.visual.transformer.layers
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    if args.amp:
        loss_scaler = NativeScaler()
        amp_autocast = torch.cuda.amp.autocast
    else:
        loss_scaler = None
        amp_autocast = suppress

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    # args.resume = True
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
    
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args=args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)
    


    start_time = time.time()
    # 早停参数
    patience = 10  # 容忍度：验证集性能未提升的最大次数
    patience_counter = 0  # 容忍度计数器
    max_accuracy = 0.0
    best_epoch = 0


    features_ema, targets, base_text_features = get_features_ema(args, eval_loader,
                                                                 model_without_ddp,
                                                                 model_ema)
    features_ema[torch.norm(features_ema, dim=1) > 1.5] /= 2  # account for the few samples that are computed twice
    features_ema = features_ema.cpu().numpy()

    # 训练之前提前确定好最优聚类数量
    K_Coarse, K_Fine = 0, 0
    # prototypes_feat = []
    # 1）Coarse
    if args.granularity in ['C', 'CF'] and args.gpu == 0:
        print("* Choose the best K for coarse prototypes")

        candidate_K = range(2, args.nb_classes // 10 + 1)
        print(candidate_K)
        cluster_results_Coarse = run_kmeans_best_k(features_ema, candidate_K, args)  # run kmeans clustering on master node
        K_Coarse = cluster_results_Coarse["best_K"]
        print("Best_K for coarse prototypes: ", K_Coarse)
        # prototypes_feat.append(cluster_results_Coarse['centroids'])
    # 2) Fine
    if args.granularity in ['F', 'CF'] and args.gpu == 0:
        print("* Choose the best K for fine prototypes")
        # 在2-4倍之间，然后进行折半查找？？
        candidate_K = [args.nb_classes * i for i in range(2, 5)]
        print(candidate_K)
        cluster_results_Fine = run_kmeans_best_k(features_ema, candidate_K, args)  # run kmeans clustering on master node
        K_Fine = cluster_results_Fine["best_K"]
        print("Best_K for fine prototypes: ", K_Fine)

        # prototypes_feat.append(cluster_results_Fine['centroids'])
    # prototypes_feat = torch.cat(prototypes_feat,0)
    # 2.细粒度
    # K_Fine = args.nb_classes * 2
    # 只对预测置信度较高的样本各自聚类，并确定每个类别的最佳聚类数量（保留高置信度样本的编号和对应的聚类数量）


    # from my_utils.utils import plot_tSNE
    # plot_tSNE(image_feat=torch.tensor(features_ema), targets=targets, prototype_feat=base_text_features,
    #           base_text_feat=cluster_results_Coarse['centroids'], text_feat=cluster_results_Fine['centroids'],
    #           classnames=None, title=args.dataset, epoch=-1, args=args)

    # del prototypes_feat


    for epoch in range(args.start_epoch, args.epochs):
        print(f"#-------------------------Clustering before training at Epoch[{epoch}]---------------#")
        cluster_result = None
        if epoch >= args.center_warmup_epochs:
            # 1. get visual ema_feature
            features_ema, targets, base_text_features = get_features_ema(args, eval_loader, model_without_ddp, model_ema)
            if args.gpu == 0:
                features_ema[
                    torch.norm(features_ema, dim=1) > 1.5] /= 2  # account for the few samples that are computed twice
                features_ema = features_ema.cpu().numpy()

                if args.granularity in ['C', 'CF']:
                    print("now is kmeans for coarse prototypes")
                    centroids_coarse, im2cluster_coarse, _ = run_kmeans(features_ema, K_Coarse, args=args)

                if args.granularity in ['F', 'CF']:
                    print("now is kmeans for fine prototypes")
                    centroids_fine, im2cluster_fine, _ = run_kmeans(features_ema, K_Fine, args=args)

                im2clusters = []
                centroids = []

                if args.granularity in ['C', 'CF']:
                    im2clusters.append(im2cluster_coarse)
                    centroids.append(centroids_coarse)

                if args.granularity in ['F', 'CF']:
                    im2clusters.append(im2cluster_fine)
                    centroids.append(centroids_fine)

                cluster_result = {'im2cluster': torch.cat(im2clusters, 0),
                                  'centroids': torch.cat(centroids, 0)
                                  }

            dist.barrier()
            # broadcast clustering result
            for _, data_tensor in cluster_result.items():
                dist.broadcast(data_tensor, 0, async_op=False)


        centroids_ema_feat = cluster_result['centroids']
        # logits_center = model_without_ddp.logit_scale.exp() * base_text_features @ centroids_ema_feat.T # [C, K]

        s_id, t_id = [], []  # 源节点s_id， 目标节点t_id

        # Coarse
        if args.granularity in ['C', 'CF']:
            # _, pseudo_targets_proto = logits_center[:, :K_Coarse].max(-1)
            # for cls_id, proto_id in enumerate(pseudo_targets_proto):
            #     s_id.append(cls_id)
            #     t_id.append(args.nb_classes + proto_id.item())
            # _, pseudo_targets_proto = logits_center[:, :K_Coarse].t().max(-1)
            # for proto_id, cls_id in enumerate(pseudo_targets_proto):
            #     s_id.append(args.nb_classes + proto_id)
            #     t_id.append(cls_id.item())
            for s in range(args.nb_classes):
                for t in range(args.nb_classes, args.nb_classes + K_Coarse):
                    s_id.append(s)
                    t_id.append(t)


        # Fine
        if args.granularity in ['F', 'CF']:
            # _, pseudo_targets_proto_fine = logits_center[:, K_Coarse:].t().max(-1)
            # for proto_id_fine, cls_id in enumerate(pseudo_targets_proto_fine):
            #     s_id.append(args.nb_classes + K_Coarse + proto_id_fine)
            #     t_id.append(cls_id)
            # _, pseudo_targets_proto_fine = logits_center[:, K_Coarse:].max(-1)
            # for cls_id, proto_id_fine in enumerate(pseudo_targets_proto_fine):
            #     t_id.append(args.nb_classes + K_Coarse + proto_id_fine)
            #     s_id.append(cls_id)
            for s in range(args.nb_classes):
                for t in range(args.nb_classes + K_Coarse, args.nb_classes + K_Coarse + K_Fine):
                    s_id.append(s)
                    t_id.append(t)


        # k = 3
        # _, topk_indices_t2p = torch.topk(logits_center, k, dim=-1)
        # s_id, t_id = [], []  # 源节点id， 目标节点id
        # for cls_id in range(topk_indices_t2p.size(0)):
        #     s_id += [cls_id] * k
        #     t_id += topk_indices_t2p[cls_id, :] + args.nb_classes



        # # 双向k-NN，取交集
        # k = 2
        # _, topk_indices_t2p = torch.topk(logits_center, k, dim=-1)
        # s_id, t_id = [], []  # 源节点id， 目标节点id
        # for cls_id in range(topk_indices_t2p.size(0)):
        #     s_id += [cls_id] * k
        #     t_id += topk_indices_t2p[cls_id, :] + len(model.module.classnames)
        # _, topk_indices_p2t = torch.topk(logits_center.T, k, dim=-1)
        # for prototype_id in range(topk_indices_p2t.size(0)):
        #     s_id += topk_indices_p2t[prototype_id, :]
        #     t_id += [prototype_id + len(model.module.classnames)]*k

        # if epoch == 0 or (epoch+1) % 10 == 0:
        #     print(s_id)
        #     print(t_id)

        #
        # # (2) 阈值法
        # 问题：原型到粗类的相似度非常非常小的时候怎么办
        # pro_center = F.softmax(logits_center, dim=-1)
        # pro_mask = pro_center > 0.1
        # indices = torch.nonzero(pro_mask)
        # prototype_id, cls_id = indices[:, 0], indices[:, 1]
        # print("number of edges:{}".format(indices.shape[0]))
        # print(f"number of prototype with edges: {prototype_id.unique().size(0)}", )
        # print(f"number of coarse classes with edges: {cls_id.unique().size(0)}", )
        # s_id = prototype_id + len(model.module.classnames)
        # t_id = cls_id
        # # 每个节点添加一个指向自身的边（自环）
        # s_id = torch.cat([s_id.to("cpu"), torch.arange(pro_center.shape[0]) + len(model.module.classnames)])
        # t_id = torch.cat([t_id.to("cpu"), torch.arange(pro_center.shape[0]) + len(model.module.classnames)])
        #

        # edges = s_id, t_id

        # # （3）k-NN
        # k = 3
        # _, topk_indices = torch.topk(logits_center, k, dim=-1)
        # s_id, t_id = [], []  # 源节点id， 目标节点id
        # for prototype_id in range(topk_indices.size(0)):
        #     s_id += [prototype_id + len(model.module.classnames)] * k
        #     t_id += topk_indices[prototype_id, :]
        # edges = torch.tensor(s_id + t_id), torch.tensor(t_id + s_id)





        # 创建无向图
        edges = torch.tensor(s_id + t_id), torch.tensor(t_id + s_id)  # 无向图
        g = dgl.graph(edges)
        # g = dgl.add_self_loop(g)  # 每个节点添加一个指向自身的边（自环）
        g = g.to(args.device)
        g = g.adjacency_matrix().to_dense() # for GCA


        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)



        train_stats = train_one_epoch(
            g, model, args, train_config,
            data_loader_train, optimizer, amp_autocast, device, epoch, loss_scaler, cluster_result,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            model_ema=model_ema,
        )

        init_proto_feat = cluster_result['centroids']

        # 验证和测试
        if args.output_dir and utils.is_main_process() and (epoch + 1) % args.eval_freq == 0:
            # 每10轮保存一番
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)

            test_stats = evaluate(data_loader_val, centroids_ema_feat, g, model, device, model_ema=model_ema, args=args)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.2f}%")
            if max_accuracy <= test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                    gap_dist, align_score, gap_dist_base, align_score_base, delta_modal = cal_gap(data_loader_train, model,
                                                                                                  device,
                                                                                                  init_proto_feat,
                                                                                                  g, epoch, args)
                    print(f" * Gap_dist_base:{gap_dist_base:.4f}, Gap_dist:{gap_dist:.4f} \n")
                    print(f" * Align_score_base:{align_score_base:.4f}, Align_score:{align_score:.4f}")
                    print(f" * delta_modal:{delta_modal:.4f}")

                patience_counter = 0  # 重置计数器
                best_epoch = epoch # 重置最优轮
            else:
                patience_counter += 1


            print(f'Max accuracy: {max_accuracy:.2f}% , {best_epoch}-th epoch')
            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="test", step=epoch)
                log_writer.update(test_ema_acc1=test_stats['ema_acc1'], head="test", step=epoch)
                log_writer.flush()
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            # if patience_counter >= patience:
            #     print(f'Early stopping at epoch {epoch}')
                # break


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    main(opts)