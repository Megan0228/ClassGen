import io
import os
import sys
import errno
import math
import time
import json
from collections import defaultdict, deque
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from timm.utils import get_state_dict

from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf
import torch.nn.functional as F

import random

from tensorboardX import SummaryWriter


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{avg:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_warmup_model(args, model_without_ddp, epoch):
    output_dir = Path(args.output_dir)

    epoch_name = str(epoch)

    checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)

    del model_without_ddp.model.visual.classifier
    to_save = {
        'model': model_without_ddp.state_dict(),
    }
    save_on_master(to_save, checkpoint_path)


""" 
def save_model(args, epoch, model, model_without_ddp, attention_model_without_ddp, optimizer,attn_optimizer, loss_scaler=None, model_ema=None, attn_model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)

    if loss_scaler is not None:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'attention_model_without_ddp': attention_model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'attn_optimizer': attn_optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }
    else:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'attention_model_without_ddp': attention_model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'attn_optimizer': attn_optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
        }

    if model_ema is not None:
        to_save['model_ema'] = get_state_dict(model_ema.ema)

    if attn_model_ema is not None:
        to_save['attn_model_ema'] = get_state_dict(attn_model_ema.ema)


    save_on_master(to_save, checkpoint_path)
"""


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler=None, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)

    if loss_scaler is not None:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }
    else:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
        }

    if model_ema is not None:
        to_save['model_ema'] = get_state_dict(model_ema.ema)

    save_on_master(to_save, checkpoint_path)


# def auto_load_model(args, model, model_without_ddp, attention_model_without_ddp, optimizer, attn_optimizer, loss_scaler=None, model_ema=None, attn_model_ema=None):
#     output_dir = Path(args.output_dir)
#
#     if args.auto_resume and len(args.resume) == 0:
#         import glob
#         all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
#         latest_ckpt = -1
#         for ckpt in all_checkpoints:
#             t = ckpt.split('-')[-1].split('.')[0]
#             if t.isdigit():
#                 latest_ckpt = max(int(t), latest_ckpt)
#         if latest_ckpt >= 0:
#             args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
#         print("Auto resume checkpoint: %s" % args.resume)
#
#     if args.resume:
#         if args.resume.startswith('https'):
#             checkpoint = torch.hub.load_state_dict_from_url(
#                 args.resume, map_location='cpu', check_hash=True)
#         else:
#             checkpoint = torch.load(args.resume, map_location='cpu')
#
#         model_without_ddp.load_state_dict(checkpoint['model'],strict=False)
#         attention_model_without_ddp.load_state_dict(checkpoint['attention_model'], strict=False)
#
#         print("Resume checkpoint %s" % args.resume)
#         if 'optimizer' in checkpoint and 'epoch' in checkpoint:
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             attn_optimizer.load_state_dict(checkpoint['attn_optimizer'])
#             args.start_epoch = checkpoint['epoch'] + 1
#             if hasattr(args, 'model_ema') and args.model_ema:
#                 model_ema._load_checkpoint(checkpoint['model_ema'])
#             if hasattr(args, 'attn_model_ema') and args.model_ema:
#                 attn_model_ema._load_checkpoint(checkpoint['attn_model_ema'])
#             if 'scaler' in checkpoint:
#                 loss_scaler.load_state_dict(checkpoint['scaler'])
#             print("With optim & sched!")
def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler=None, model_ema=None):
    output_dir = Path(args.output_dir)

    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if hasattr(args, 'model_ema') and args.model_ema:
                model_ema._load_checkpoint(checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)
    return torch.cat(tensor_all, dim=0)


from sklearn import metrics


def mean_per_class(outputs, targets):
    pred = outputs.argmax(1)
    confusion_matrix = metrics.confusion_matrix(targets, pred)
    per_classes = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

    return 100 * per_classes.mean()


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def cal_gap_newVersion(data_loader, model, device: torch.device, init_proto_feat, graph, epoch: int = -1, args = None, is_plot=False):
    # switch to evaluate mode
    model.eval()

    base_text_feat = model.module.base_text_features
    delta_modal = 0
    align_score_sum = 0
    align_score_sum_base = 0
    with torch.no_grad():
        all_img_feat = []
        all_target = []
        for step, (((_, images_strong, _), target), image_id) in enumerate(data_loader):
            images_strong = images_strong.to(device)
            target = target.to(device)
            logits, batch_img_feat, proto_feat, text_feat = model(images_strong, init_proto_feat, graph)

            all_img_feat.append(batch_img_feat)
            all_target.append(target)

            align_score_sum += torch.mean(torch.norm(batch_img_feat - text_feat[target], p=2, dim=-1))
            align_score_sum_base += torch.mean(torch.norm(batch_img_feat - base_text_feat[target], p=2, dim=-1))

            logit_scale = model.module.logit_scale.exp()
            delta_modal = delta_modal + torch.mean(torch.sqrt(2 - 2 * logits / logit_scale))



        all_target = torch.cat(all_target, dim=0)
        all_img_feat = torch.cat(all_img_feat, dim=0)
        delta_modal = delta_modal / (step+1)

        # base_text_centroid
        base_text_centroid = F.normalize(base_text_feat.mean(dim=0), dim=-1)

        # text_centroid
        text_centroid = F.normalize(text_feat.mean(dim=0), dim=-1)

        # img_centroid
        img_centroid = F.normalize(all_img_feat.mean(dim=0), dim=-1)

        # Gap
        gap_dist = torch.norm(img_centroid - text_centroid, p=2)
        gap_dist_base = torch.norm(img_centroid - base_text_centroid, p=2)

        align_score = align_score_sum / (step + 1)
        align_score_base = align_score_sum_base / (step + 1)

    delta_modal = delta_modal.item()
    # if epoch == 0 or (epoch + 1) % 5 == 0:
    if is_plot:
        title = f"Epoch[{epoch}]: delta_modal: {delta_modal:.4f}"
        print(title)
        plot_tSNE(image_feat=all_img_feat, text_feat=text_feat, classnames=model.module.classnames, title=None, epoch=epoch, args=args)

    return delta_modal


def cal_gap(data_loader, model, device: torch.device, init_proto_feat, graph, epoch: int = -1, args = None):
    # switch to evaluate mode
    model.eval()

    base_text_feat = model.module.base_text_features

    align_score_sum = 0
    align_score_sum_base = 0
    delta_modal = 0
    with torch.no_grad():
        all_img_feat = []
        all_target = []
        for step, (((_, images_strong, _), target), image_id) in enumerate(data_loader):
            images_strong = images_strong.to(device)
            target = target.to(device)
            logits, batch_img_feat, proto_feat, text_feat = model(images_strong, init_proto_feat, graph)

            all_img_feat.append(batch_img_feat)
            all_target.append(target)

            align_score_sum += torch.mean(torch.norm(batch_img_feat - text_feat[target], p=2, dim=-1))
            align_score_sum_base += torch.mean(torch.norm(batch_img_feat - base_text_feat[target], p=2, dim=-1))

            logit_scale = model.module.logit_scale.exp()
            delta_modal = delta_modal + torch.mean(torch.sqrt(2 - 2 * logits / logit_scale))

        all_target = torch.cat(all_target, dim=0)
        all_img_feat = torch.cat(all_img_feat, dim=0)

        # base_text_centroid
        base_text_centroid = F.normalize(base_text_feat.mean(dim=0), dim=-1)

        # text_centroid
        text_centroid = F.normalize(text_feat.mean(dim=0), dim=-1)

        # img_centroid
        img_centroid = F.normalize(all_img_feat.mean(dim=0), dim=-1)

        # Gap
        gap_dist = torch.norm(img_centroid - text_centroid, p=2)
        gap_dist_base = torch.norm(img_centroid - base_text_centroid, p=2)

        align_score = align_score_sum / (step + 1)
        align_score_base = align_score_sum_base / (step + 1)
        delta_modal = delta_modal / (step + 1)


    gap_dist, align_score, gap_dist_base, align_score_base = gap_dist.item(), align_score.item(), gap_dist_base.item(), align_score_base.item()
    delta_modal = delta_modal.item()

    if epoch == 0 or (epoch + 1) % 5 == 0:
        title = f"Epoch[{epoch}]: Gap dist: {gap_dist:.4f}, align dist:{align_score:.4f}, delta_modal:{delta_modal:.4f}"
        print(title)
        # plot_tSNE(all_img_feat, all_target, proto_feat, base_text_feat, text_feat=text_feat, classnames=model.module.classnames, title=None, epoch=epoch, args=args)
        plot_tSNE(image_feat=all_img_feat, targets=all_target, text_feat=text_feat, classnames=model.module.classnames, title=None, epoch=epoch, args=args)

    return gap_dist, align_score, gap_dist_base, align_score_base, delta_modal


def plot_tSNE(image_feat=None, targets=None, prototype_feat=None, base_text_feat=None, text_feat=None, classnames=None, title=None, epoch=-1, args=None):
    """
    args:
    targets, classnames,
    """

    # 使用t-SNE进行降维

    # plot_dir = f"/public/gsp/code/UHS_GAT/Plot/{args.granularity}/{args.dataset}"


    combined_features=[]
    if image_feat is not None:
        combined_features.append(image_feat.to("cpu").detach().numpy())
    if prototype_feat is not None:
        combined_features.append(prototype_feat.to("cpu").detach().numpy())
    if base_text_feat is not None:
        combined_features.append(base_text_feat.to("cpu").detach().numpy())
    if text_feat is not None:
        combined_features.append(text_feat.to("cpu").detach().numpy())
    combined_features = np.concatenate(combined_features, axis=0)


    if targets is not None:
        targets = targets.to("cpu").detach().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(combined_features)

    color_map = plt.cm.get_cmap('tab20', 10)
    # colors = color_map(np.arange(10))


    # 可视化结果
    plt.figure(figsize=(10, 10))

    start = 0

    # Image Instances
    if image_feat is not None:
        start = plt_scatter(image_feat, reduced_features, 'blue', 'o', 50, start)
        # n_img = image_feat.shape[0]
        # plt.scatter(reduced_features[start:start + n_img, 0], reduced_features[start:start + n_img, 1],
        #             c='blue',
        #             marker='o',
        #             s=50
        #             )
        # start += n_img

    # Prototypes
    if prototype_feat is not None:
        start = plt_scatter(prototype_feat, reduced_features, 'black', '*', 150, start)
        # n_proto = prototype_feat.shape[0]
        # plt.scatter(reduced_features[start:start + n_proto, 0], reduced_features[start:start + n_proto, 1],
        #             c='black',
        #             marker='*',
        #             s=100
        #             )
        # start += n_proto

    # 初始文本特征
    if base_text_feat is not None:
        start = plt_scatter(base_text_feat, reduced_features, 'orange', '*', 100, start)
        # plt.scatter(reduced_features[n_img + n_proto:n_img + n_proto + n_t, 0],
        #             reduced_features[n_img + n_proto:n_img + n_proto + n_t, 1],
        #             c='none',
        #             edgecolors='green',
        #             marker='*',
        #             s=150
        #             )

        # 绘制更新后的文本特征
    if text_feat is not None:
        start = plt_scatter(text_feat, reduced_features, 'red', '*', 150, start)
        # plt.scatter(reduced_features[n_img + n_proto + n_t:, 0], reduced_features[n_img + n_proto + n_t:, 1],
        #             c='black',
        #             marker='*',
        #             s=150
        #             )



    if title is not None:
        plt.title(title)

    plot_dir = os.path.join(args.output_dir, "tSNE")
    mkdir_if_missing(plot_dir)
    save_path = plot_dir + f"/Epoch[{epoch}].png"
    plt.savefig(save_path)


def plt_scatter(feats, reduced_features, color, marker, size, start):
    steps = feats.shape[0]
    plt.scatter(reduced_features[start:start + steps, 0], reduced_features[start:start + steps, 1],
                c=color,
                marker=marker,
                s=size
                )
    start += steps
    return start




