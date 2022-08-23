import torch
import torch.nn as nn
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys

from classification import config_parser, _parse_args, _logger, train_one_epoch, validate
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from importlib import import_module

from PIL import Image

from src.classification import *

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass
import random

class FedCIAFR100(Dataset):
    def __init__(self, path, idx_per_worker=50, total_worker=5, worker_idx=0, log_prefix='')->None:
        dataset = h5py.File(path, 'r')
        total_idx = len(dataset['examples'].keys())
        assert idx_per_worker * total_worker <= total_idx and worker_idx < total_worker
        all_idx = random.sample(sorted(dataset['examples'].keys()), k=idx_per_worker * total_worker)
        local_idx = all_idx[worker_idx * idx_per_worker: (worker_idx + 1) * idx_per_worker]
        # _logger.info(f'{log_prefix} idx: {local_idx}')
        self.x = np.vstack([dataset['examples'][idx]['image'][()] for idx in local_idx])
        self.y = np.vstack([dataset['examples'][idx]['label'][()][:, None] for idx in local_idx]).squeeze()
        self.transform = None
        self.target_transform = None
    
    def __len__(self)->int:
        return len(self.x)

    def __getitem__(self, index) -> tuple:
        img, target = self.x[index], self.y[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

setup_default_logging()
writer = SummaryWriter('./adapt_log')
sys.argv = ['', "-c", "Convolutional-MLPs/configs/classification/finetuned/convmlp_m_cifar100.yml", "--resume", "Convolutional-MLPs/chkpt/convmlp_m_cifar100.pth", "--download", "--data_dir", "Convolutional-MLPs/dataset"]

args, args_text = _parse_args()

args.prefetcher = not args.no_prefetcher
args.distributed = False
args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
args.world_size = 1
args.rank = 0
use_amp = None
if args.amp:
    # `--amp` chooses native amp before apex (APEX ver not actively maintained)
    if has_native_amp:
        args.native_amp = True
    elif has_apex:
        args.apex_amp = True
if args.apex_amp and has_apex:
    use_amp = 'apex'
elif args.native_amp and has_native_amp:
    use_amp = 'native'
elif args.apex_amp or args.native_amp:
    _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                    "Install NVIDA apex or upgrade to PyTorch 1.6")
random_seed(args.seed)

model = create_model(
    args.model,
    pretrained=args.pretrained,
    num_classes=args.num_classes,
    drop_rate=args.drop,
    drop_connect_rate=args.drop_connect,  # DEPRECATED, usedrop_path
    drop_path_rate=args.drop_path,
    drop_block_rate=args.drop_block,
    global_pool=args.gp,
    bn_tf=args.bn_tf,
    bn_momentum=args.bn_momentum,
    bn_eps=args.bn_eps,
    scriptable=args.torchscript,
    checkpoint_path=args.initial_checkpoint)

param_count = sum([m.numel() for m in model.parameters()])
data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

num_aug_splits = 0
if args.aug_splits > 0:
    assert args.aug_splits > 1, 'A split of 1 makes no sense'
    num_aug_splits = args.aug_splits

# enable split bn (separate bn stats per batch-portion)
if args.split_bn:
    assert num_aug_splits > 1 or args.resplit
    model = convert_splitbn_model(model, max(num_aug_splits, 2))

model.to(args.device)
optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
amp_autocast = None  # do nothing
loss_scaler = None
if use_amp == 'apex':
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    loss_scaler = ApexScaler()
    if args.local_rank == 0:
        _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
elif use_amp == 'native':
    amp_autocast = torch.cuda.amp.autocast
    loss_scaler = NativeScaler()
    if args.local_rank == 0:
        _logger.info('Using native Torch AMP. Training in mixed precision.')
else:
    if args.local_rank == 0:
        _logger.info('AMP not enabled. Training in float32.')

resume_epoch = None
if args.resume:
    resume_epoch = resume_checkpoint(
        model, args.resume,
        optimizer=None if args.no_resume_opt else optimizer,
        loss_scaler=None if args.no_resume_opt else loss_scaler,
        log_info=args.local_rank == 0)

model_ema = None
if args.model_ema:
    # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    model_ema = ModelEmaV2(
        model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
    if args.resume:
        load_checkpoint(model_ema.module, args.resume, use_ema=True)

lr_scheduler, num_epochs = create_scheduler(args, optimizer)
start_epoch = 0
if args.start_epoch is not None:
    # a specified start_epoch will always override the resume epoch
    start_epoch = args.start_epoch
elif resume_epoch is not None:
    start_epoch = resume_epoch
if lr_scheduler is not None and start_epoch > 0:
    lr_scheduler.step(start_epoch)


# setup mixup / cutmix
collate_fn = None
mixup_fn = None
mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
if mixup_active:
    mixup_args = dict(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing, num_classes=args.num_classes)
    if args.prefetcher:
        assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
        collate_fn = FastCollateMixup(**mixup_args)
    else:
        mixup_fn = Mixup(**mixup_args)

noise = 'image_blur_'
dataset_train = FedCIAFR100(f'Convolutional-MLPs/datasets/{noise}fed_cifar100_train.h5', total_worker=1, idx_per_worker=500, worker_idx=0)
dataset_eval = FedCIAFR100(f'Convolutional-MLPs/datasets/{noise}fed_cifar100_test.h5', total_worker=1, idx_per_worker=100, worker_idx=0)
dataset_eval_orig = FedCIAFR100(f'Convolutional-MLPs/datasets/fed_cifar100_test.h5', total_worker=1, idx_per_worker=100, worker_idx=0)
# wrap dataset in AugMix helper
if num_aug_splits > 1:
    dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

# create data loaders w/ augmentation pipeiine
train_interpolation = args.train_interpolation
if args.no_aug or not train_interpolation:
    train_interpolation = data_config['interpolation']
loader_train = create_loader(
    dataset_train,
    input_size=data_config['input_size'],
    batch_size=args.batch_size,
    is_training=True,
    use_prefetcher=args.prefetcher,
    no_aug=args.no_aug,
    re_prob=args.reprob,
    re_mode=args.remode,
    re_count=args.recount,
    re_split=args.resplit,
    scale=args.scale,
    ratio=args.ratio,
    hflip=args.hflip,
    vflip=args.vflip,
    color_jitter=args.color_jitter,
    auto_augment=args.aa,
    num_aug_splits=num_aug_splits,
    interpolation=train_interpolation,
    mean=data_config['mean'],
    std=data_config['std'],
    num_workers=args.workers,
    distributed=args.distributed,
    collate_fn=collate_fn,
    pin_memory=args.pin_mem,
    use_multi_epochs_loader=args.use_multi_epochs_loader
)

loader_eval = create_loader(
    dataset_eval,
    input_size=data_config['input_size'],
    batch_size=args.validation_batch_size_multiplier * args.batch_size,
    is_training=False,
    use_prefetcher=args.prefetcher,
    interpolation=data_config['interpolation'],
    mean=data_config['mean'],
    std=data_config['std'],
    num_workers=args.workers,
    distributed=args.distributed,
    crop_pct=data_config['crop_pct'],
    pin_memory=args.pin_mem,
)

loader_eval_orig = create_loader(
    dataset_eval_orig,
    input_size=data_config['input_size'],
    batch_size=args.validation_batch_size_multiplier * args.batch_size,
    is_training=False,
    use_prefetcher=args.prefetcher,
    interpolation=data_config['interpolation'],
    mean=data_config['mean'],
    std=data_config['std'],
    num_workers=args.workers,
    distributed=args.distributed,
    crop_pct=data_config['crop_pct'],
    pin_memory=args.pin_mem,
)

# setup loss function
if args.jsd:
    assert num_aug_splits > 1  # JSD only valid with aug splits set
    train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
elif mixup_active:
    # smoothing is handled with mixup target transform
    train_loss_fn = SoftTargetCrossEntropy().cuda()
elif args.smoothing:
    train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
else:
    train_loss_fn = nn.CrossEntropyLoss().cuda()
validate_loss_fn = nn.CrossEntropyLoss().cuda()

# setup checkpoint saver and eval metric tracking
eval_metric = args.eval_metric
best_metric = None
best_epoch = None
saver = None
output_dir = None
if args.rank == 0:
    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            safe_model_name(args.model),
            str(data_config['input_size'][-1])
        ])
    output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
    decreasing = True if eval_metric == 'loss' else False
    saver = CheckpointSaver(
        model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
        checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

validate(model, loader_eval_orig, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix='Original')
eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix='Noised')
writer.add_scalar('ConvMLP/test_accuracy', eval_metrics['top1'], 0)

for epoch in range(start_epoch, num_epochs):
    # train_metrics = OrderedDict([('loss', 0.)])
    train_metrics = train_one_epoch(
        epoch, model, loader_train, optimizer, train_loss_fn, args,
        lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
        amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)

    eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix='Adapted')

    writer.add_scalar('ConvMLP/test_accuracy', eval_metrics['top1'], epoch + 1)
    if model_ema is not None and not args.model_ema_force_cpu:
        ema_eval_metrics = validate(
            model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast,
            log_suffix=' (EMA)')
        eval_metrics = ema_eval_metrics

    if lr_scheduler is not None:
        # step LR for next epoch
        lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

    if output_dir is not None:
        update_summary(
            epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
            write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

    if saver is not None:
        # save proper checkpoint with eval metric
        save_metric = eval_metrics[eval_metric]
        best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))