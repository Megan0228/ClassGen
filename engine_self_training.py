import pdb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from my_utils.utils import mkdir_if_missing

import math
import sys
from typing import Iterable
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_utils import utils
from my_utils.utils import cal_gap
from timm.utils import accuracy


def train_one_epoch(graph, model: torch.nn.Module, args, train_config,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, amp_autocast,
                    device: torch.device, epoch: int, loss_scaler, cluster_result=None,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, model_ema=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')


    for step, (((images_weak, images_strong, mask), targets), image_id) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]

        # ramp-up ema decay
        model_ema.decay = train_config['model_ema_decay_init'] + (
                args.model_ema_decay - train_config['model_ema_decay_init']) * min(1, it / train_config['warm_it'])
        metric_logger.update(ema_decay=model_ema.decay)

        images_weak, images_strong = images_weak.to(device, non_blocking=True), images_strong.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        init_proto_feat = cluster_result['centroids']
        with torch.no_grad():
            # pseudo-label with ema model(weak aug): hard probability, one-hot vector
            # logits_ema, batch_feature_w, proto_feat_ema, text_feat_ema = model_ema.ema(images_weak, init_proto_feat, graph)
            ##

            logits_ema, batch_feature_w, proto_feat_ema, text_feat_ema = model_ema.ema(images_weak, init_proto_feat, graph)
            # ----pseudo label with graph------#
            probs_ema = F.softmax(logits_ema, dim=-1)
            score, pseudo_targets = probs_ema.max(-1)
            conf_mask = score > train_config['conf_threshold']
            pseudo_label_acc = (pseudo_targets[conf_mask] == targets[conf_mask]).float().mean().item()
            conf_ratio = conf_mask.float().sum() / conf_mask.size(0)
            metric_logger.update(conf_ratio=conf_ratio)
            metric_logger.update(pseudo_label_acc=pseudo_label_acc)

            # ----pseudo label without graph------#
            logit_scale = model_ema.ema.logit_scale
            logits_ema_wo = logit_scale.exp() * batch_feature_w @ model_ema.ema.base_text_features.T
            probs_ema_wo = F.softmax(logits_ema_wo, dim=-1)
            score_wo, pseudo_targets_wo = probs_ema_wo.max(-1)
            conf_mask_wo = score_wo > train_config['conf_threshold']  # 高于阈值，样本筛选
            pseudo_label_acc_wo = (pseudo_targets_wo[conf_mask_wo] == targets[conf_mask_wo]).float().mean().item()
            metric_logger.update(pseudo_label_acc_wo=pseudo_label_acc_wo)

            # # # ----coarse pseudo label------#
            # logits_ema_coarse = logit_scale.exp() * batch_feature_w @ proto_feat_ema.T
            # probs_ema_coarse = F.softmax(logits_ema_coarse, dim=-1)
            # score_coarse, pseudo_targets_coarse = probs_ema_coarse.max(-1)
            # conf_mask_coarse = score_coarse > train_config['conf_threshold']  # 高于阈值，样本筛选
            # # -----------------------------#


            # score_ema_xz, pseudo_targets_xz = prob_xz_ema.max(-1, keepdim=True)
            # pseudo_score_xz = torch.zeros_like(prob_xz_ema)

            # pseudo_score_xz.scatter_(1, pseudo_targets_xz, score_ema_xz)
            # pseudo_targets_xz = F.one_hot(pseudo_targets_xz, proto_feat_ema.shape[0])  # b, k
            # pseudo_targets_xz = pseudo_targets_xz.type(model_ema.ema.dtype)

            # logits_proto_cls_ema = logit_scale.exp() * proto_feat_ema @ text_feat_ema.T
            # prob_zc_ema = F.softmax(logits_proto_cls_ema, dim=-1)
            # score_ema_zc, pseudo_targets_zc = prob_zc_ema.max(-1)
            # # score_ema_zc, pseudo_targets_zc = prob_zc_ema.max(-1, keepdim=True)
            # # pseudo_score_zc = torch.zeros_like(prob_zc_ema)
            # # pseudo_score_zc.scatter_(1, pseudo_targets_zc, score_ema_zc)
            # # pseudo_targets_zc = F.one_hot(pseudo_targets_zc, text_feat_ema.shape[0])  # k, c
            # # pseudo_targets_zc = pseudo_targets_zc.type(model_ema.ema.dtype)
            #
            # # pseudo_targets_hier = pseudo_score_xz @ pseudo_score_zc
            # probs_ema_hier = prob_xz_ema @ prob_zc_ema
            # score_hier, pseudo_targets_hier = probs_ema_hier.max(-1)
            #
            # conf_mask_hier = score_hier > train_config['conf_threshold']  # 高于阈值，样本筛选
            # pseudo_label_acc_hier = (pseudo_targets_hier[conf_mask_hier] == targets[conf_mask_hier]).float().mean().item()
            # conf_ratio_hier = conf_mask_hier.float().sum() / conf_mask_hier.size(0)
            # metric_logger.update(pseudo_label_acc_hier=pseudo_label_acc_hier)
            # metric_logger.update(conf_ratio_hier=conf_ratio_hier)


            # probs_ema_hybrid = 0.5 * probs_ema_hier + 0.5 * probs_ema
            # score_hybrid, pseudo_targets_hybrid = probs_ema_hybrid.max(-1)
            #
            # conf_mask_hybrid = score_hybrid > train_config['conf_threshold']  # 高于阈值，样本筛选
            # pseudo_label_acc_hybrid = (
            #             pseudo_targets_hybrid[conf_mask_hybrid] == targets[conf_mask_hybrid]).float().mean().item()
            # conf_ratio_hybrid = conf_mask_hybrid.float().sum() / conf_mask_hybrid.size(0)
            # metric_logger.update(pseudo_label_acc_hybrid=pseudo_label_acc_hybrid)
            # metric_logger.update(conf_ratio_hybrid=conf_ratio_hybrid)



        with amp_autocast():
            # ----prediction ------#
            if args.mask:
                logits, batch_feature_s, x_mask, x_recon, mask, w, proto_feat, text_feat = model(images_strong, init_proto_feat, graph, mask=mask)
            else:
                # print("without masking")
                logits, batch_feature_s, proto_feat, text_feat = model(images_strong, init_proto_feat, graph)


            logit_scale = model.module.logit_scale.exp()
            # logits_img_proto = logit_scale * batch_feature_s @ proto_feat.T
            # logits_proto_cls = logit_scale * proto_feat @ text_feat.T


            # 1. self-training loss
            # loss_st_fine = F.cross_entropy(logits_img_proto, pseudo_targets_xz)
            # loss_st_coarse = F.cross_entropy(logits_proto_cls, pseudo_targets_zc)


            loss_st = F.cross_entropy(logits[conf_mask], pseudo_targets[conf_mask])
            # loss_st_ori = F.cross_entropy(logits[conf_mask], pseudo_targets_wo[conf_mask])

            # 2. fairness regularization of self-training
            probs = F.softmax(logits, dim=-1)
            probs_all = utils.all_gather_with_grad(probs)
            probs_batch_avg = probs_all.mean(0)  # average prediction probability across all gpus
            if args.nb_classes >= 512:
                # moving average
                if step == 0:
                    probs_avg = probs_batch_avg
                else:
                    probs_avg = 0.5 * (probs_avg.detach() + probs_batch_avg)
                loss_fair = -(torch.log(probs_avg)).mean() / 0.5
            else:
                # batch average
                probs_avg = probs_batch_avg
                loss_fair = -(torch.log(probs_avg)).mean()



            # 3. regularization
            loss_reg = torch.mean(torch.sqrt(2 - 2 * logits / logit_scale))
            # pos_sim = logits.gather(1, pseudo_targets.unsqueeze(1)) / logit_scale
            # loss_reg = torch.mean(2 - 2 * pos_sim)
            # pdb.set_trace()


            # ----coarse prediction------#
            # logits_coarse = logit_scale.exp() * batch_feature_s @ proto_feat.T
            # loss_st_coarse = F.cross_entropy(logits_coarse[conf_mask_coarse], pseudo_targets_coarse[conf_mask_coarse])
            # # fairness regularization of self-training
            # probs_coarse = F.softmax(logits_coarse, dim=-1)
            # probs_all_coarse = utils.all_gather_with_grad(probs_coarse)
            # probs_batch_avg_coarse = probs_all_coarse.mean(0)  # average prediction probability across all gpus
            # if args.nb_classes >= 512:
            #     # moving average
            #     if step == 0:
            #         probs_avg_coarse = probs_batch_avg_coarse
            #     else:
            #         probs_avg_coarse = 0.5 * (probs_avg_coarse.detach() + probs_batch_avg_coarse)
            #     loss_fair_coarse = -(torch.log(probs_avg_coarse)).mean() / 0.5
            # else:
            #     # batch average
            #     probs_avg_coarse = probs_batch_avg_coarse
            #     loss_fair_coarse = -(torch.log(probs_avg_coarse)).mean()



            loss = 0
            loss_align_patch = 0
            if cluster_result is not None:
                im2cluster = cluster_result['im2cluster']
                prototypes = cluster_result['centroids']
                pos_proto_id = im2cluster[image_id]
                # pos_prototypes = prototypes[pos_proto_id]
                pos_prototypes = proto_feat_ema[pos_proto_id]
                # proto_center, centroids = pos_prototypes, prototypes

                if args.mask:
                    loss_align = torch.sum((x_mask - pos_prototypes.unsqueeze(1)).pow(2), dim=-1, keepdim=True)
                    loss_align = loss_align[w.bool()].view(w.size(0), -1)
                    loss_align = torch.mean(loss_align)
                    loss_align_patch += loss_align

            # if args.mask:
            #     loss_mim = F.l1_loss(x_recon, images_strong, reduction='none')
            #     loss_mim = (loss_mim * mask).sum() / mask.sum() / images_strong.size(1)
            #     loss += args.l1 * loss_mim + train_config['w_align'] * loss_align_patch + args.l3 * (loss_st + train_config['w_fair'] * loss_fair )
            # else:

            loss += loss_st + train_config['w_fair'] * loss_fair + args.wreg * loss_reg


        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        if loss_scaler is not None:
            grad_norm = loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            metric_logger.update(loss_scale=loss_scale_value)
            metric_logger.update(grad_norm=grad_norm)
        else:
            loss.backward(create_graph=False)
            optimizer.step()

        model_ema.update(model)
        torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_st=loss_st.item())
        metric_logger.update(loss_reg=loss_reg.item())
        # metric_logger.update(loss_st_coarse=loss_st_coarse.item())
        metric_logger.update(loss_fair=loss_fair.item())
        # metric_logger.update(loss_fair_coarse=loss_fair_coarse.item())
        # metric_logger.update(loss_st_fine=loss_st_fine.item())
        # metric_logger.update(loss_hst=loss_hst.item())
        # metric_logger.update(loss_hyst=loss_hyst.item())
        # if args.mask:
        #     # metric_logger.update(loss_mim=loss_mim.item())
        #     metric_logger.update(loss_align_patch=loss_align_patch.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        if log_writer is not None:
            log_writer.update(loss=loss.item(), head="train")
            log_writer.update(loss_st=loss_st.item(), head="train")
            log_writer.update(loss_reg=loss_reg.item(), head="train")
            # log_writer.update(loss_st_coarse=loss_st_coarse.item(), head="train")
            log_writer.update(loss_fair=loss_fair.item(), head="train")
            # log_writer.update(loss_fair_coarse=loss_fair_coarse.item(), head="train")
            # log_writer.update(loss_st_fine=loss_st_fine.item(), head="train")
            # log_writer.update(loss_hyst=loss_hyst.item(), head="train")
            # if args.mask:
            #     log_writer.update(loss_mim=loss_mim.item(), head="train")
            #     log_writer.update(loss_align_patch=loss_align_patch.item(), head="train")

            log_writer.update(conf_ratio=conf_ratio, head="train")
            # log_writer.update(conf_ratio_hier=conf_ratio_hier, head="train")
            # log_writer.update(conf_ratio_hybrid=conf_ratio_hybrid, head="train")
            log_writer.update(pseudo_label_acc=pseudo_label_acc, head="train")
            log_writer.update(pseudo_label_acc_wo=pseudo_label_acc_wo, head="train")
            # log_writer.update(pseudo_label_acc_hier=pseudo_label_acc_hier, head="train")
            # log_writer.update(pseudo_label_acc_hybrid=pseudo_label_acc_hybrid, head="train")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Calculate GAP

    gap_dist, align_score, gap_dist_base, align_score_base, delta_modal = cal_gap(data_loader, model, device, init_proto_feat, graph, epoch, args)
    print(f" * Gap_dist_base:{gap_dist_base:.4f}, Gap_dist:{gap_dist:.4f} \n")
    print(f" * Align_score_base:{align_score_base:.4f}, Align_score:{align_score:.4f}")
    print(f" * delta_modal:{delta_modal:.4f}" )
    # log_writer.update(gap_dist=gap_dist, head="train", step=epoch)
    # log_writer.update(gap_dist_base=gap_dist_base, head="train", step=epoch)
    # log_writer.update(align_score=align_score, head="train", step=epoch)
    # log_writer.update(align_score_base=align_score_base, head="train", step=epoch)


    """
     # 计算模态GAP 重新计算
    with torch.no_grad():
        # image centroid
        image_feat_centroid = F.normalize(image_feat_centroid/(step+1), dim=-1)

        # base text
        base_text_feat = model.module.base_text_features
        base_text_feat_centroid = F.normalize(base_text_feat.mean(dim=0), dim=-1)
        gap_base = torch.norm(image_feat_centroid-base_text_feat_centroid, p=2)
        print("Gap base: ", gap_base.item())


        # updated text
        updated_text_feat_centroid = F.normalize(text_feat.mean(dim=0), dim=-1)
        gap_new = torch.norm(image_feat_centroid - updated_text_feat_centroid, p=2)
        print("Gap updated: ", gap_new.item())

    # plot two figures 包括图像实例，文本特征和聚类中心
    if epoch == 0 or (epoch+1) % 10 == 0:
        plot_tsne(images_feat_set, targets_set, proto_feat, base_text_feat, text_feat, model.module.classnames,
                  f"Epoch_{epoch}: Gap base: {gap_base:.4f}, Gap New: {gap_new:.4f}",
                  epoch, args.dataset
                  )
    """

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy_proto(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def evaluate(data_loader, centroids_ema_feat, graph, model, device, model_ema=None, args=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if model_ema is not None:
        model_ema.ema.eval()

    if args.dataset in ['pets', 'caltech101']:
        all_outputs = []
        all_ema_outputs = []
        all_targets = []
        all_outputs_without_graph = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        data = batch[0]
        images = data[0].to(device, non_blocking=True)
        target = data[-1].to(device, non_blocking=True)
        image_id = batch[1]

        # compute output
        output, _, _, text_feat = model(images, centroids_ema_feat, graph)
        output_without_graph, _ = model(images)


        # Accu@1
        if args.dataset in ['pets', 'caltech101']:
            all_outputs.append(output.cpu())
            all_targets.append(target.cpu())
            all_outputs_without_graph.append(output_without_graph.cpu())
        else:
            acc = accuracy(output, target)[0]
            metric_logger.meters['acc1'].update(acc.item(), n=images.shape[0])

            acc_without_graph = accuracy(output_without_graph, target)[0]
            metric_logger.meters['acc1_wg'].update(acc_without_graph.item(), n=images.shape[0])


        if model_ema is not None:
            ema_output, _, _, _ = model_ema.ema(images, centroids_ema_feat, graph)

            if args.dataset in ['pets', 'caltech101']:
                all_ema_outputs.append(ema_output.cpu())
            else:
                ema_acc1 = accuracy(ema_output, target)[0]
                metric_logger.meters['ema_acc1'].update(ema_acc1.item(), n=images.shape[0])

    if args.dataset in ['pets', 'caltech101']:
        mean_per_class = utils.mean_per_class(torch.cat(all_outputs), torch.cat(all_targets))
        metric_logger.meters['acc1'].update(mean_per_class)

        mean_per_class_without_graph = utils.mean_per_class(torch.cat(all_outputs_without_graph), torch.cat(all_targets))
        metric_logger.meters['acc1_wg'].update(mean_per_class_without_graph)

        if model_ema is not None:
            mean_per_class = utils.mean_per_class(torch.cat(all_ema_outputs), torch.cat(all_targets))
            metric_logger.meters['ema_acc1'].update(mean_per_class)

    print('* Acc@1 {top1.global_avg:.2f}'.format(top1=metric_logger.acc1))
    print('* Acc@1 {top1.global_avg:.2f} without graph'.format(top1=metric_logger.acc1_wg))

    # Modality Gap


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    

