import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.utils.data import Dataset
from math import floor

import sys

sys.path.insert(0, 'Convolutional-MLPs')
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

from torch.utils.data import Dataset, DataLoader, Subset, random_split

from PIL import Image

from src.classification import *

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.data.mixup import one_hot, mixup_target, rand_bbox, rand_bbox_minmax, cutmix_bbox_and_lam
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

from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import InterpolationMode


class Mixup:
    """ Mixup/Cutmix that applies different params to each element or whole batch

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def _params_per_elem(self, batch_size):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=np.bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size, dtype=np.bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_pair(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    x[j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, device=x.device)
        return x, target

def dataset_noniid(
    img,
    targets,
    num_shards: int = 4,
    client_id: int = 0,
    num_clients: int = 0,
):
    """Load non-IID Dataset (training and test set)."""
    num_imgs = len(img) // num_shards
    num_clients = max(num_clients, 1)
    shards_per = num_shards // num_clients
    # Partition data
    shard_idxs = np.arange(num_shards)
    sample_idxs = np.arange(num_imgs * num_shards)
    sample_labels = np.array(targets[:num_imgs * num_shards])

    # Sort
    sample_pairs = np.vstack((sample_idxs, sample_labels))
    sample_pairs = sample_pairs[:, sample_pairs[1, :].argsort()]
    sample_idxs = sample_pairs[0, :]

    # Divide and assign
    # np.random.shuffle(shard_idxs)
    client_shards = shard_idxs[client_id * shards_per : (client_id + 1) * shards_per]
    client_samples = np.concatenate(
        tuple(
            sample_idxs[x * num_imgs : x * num_imgs + num_imgs]
            for x in np.nditer(client_shards)
        )
    )
    img = img[client_samples]
    targets = np.array(targets)[client_samples]

    return img, targets

class FedCIAFR100(VisionDataset):
    def __init__(self, path, num_shards, client_id, num_client,
                 log_prefix='', train=True, transform=None, target_transform=None) -> None:
        
        super(FedCIAFR100, self).__init__(
            '', transform=transform,
            target_transform=target_transform)
        
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        all_idx = list(dataset['images'].keys())
        img = np.vstack([dataset['images'][idx] for idx in all_idx])
        targets = np.vstack([dataset['labels'][idx][:, None] for idx in all_idx]).squeeze()
        if client_id < 0:
            client_id = 0
        img, targets = dataset_noniid(img, targets, num_shards, client_id, num_client)
        # local_idx_num = len(all_idx) // num_shards
        # local_idx = all_idx[client_id*local_idx_num : client_id*local_idx_num + local_idx_num]
        # img = np.vstack([dataset['images'][idx] for idx in local_idx])
        # targets = np.vstack([dataset['labels'][idx][:, None] for idx in local_idx]).squeeze()
        self.x = img
        self.y = targets
        print(set(self.y))

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

class ConvMLP:
    def __init__(self, worker_idx=0, batch_size=25, 
                 num_shards=4, num_client=4, device=None):
        temp_arg = ["-c", "Convolutional-MLPs/configs/classification/finetuned/convmlp_m_cifar100.yml", "--resume", "Convolutional-MLPs/chkpt/convmlp_m_cifar100.pth", "--download", "--seed", "32", "-j", "1", "--data_dir", "./datasets ", "--no-prefetcher"]

        args, args_text = _parse_args(temp_arg)

        args.prefetcher = not args.no_prefetcher
        args.distributed = False
        args.device = device
        if device is None:
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
            pretrained=False,
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

        # Count number of parameters
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

        # model.to(args.device)
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

        args.batch_size = batch_size

        # noise = 'image_blur_'
        noise = 'mixed_'
        dataset_train = FedCIAFR100(f'datasets/{noise}fed_cifar100_train.pkl', num_shards, worker_idx, num_client, train=True,log_prefix='Train ')
        dataset_eval = FedCIAFR100(f'datasets/{noise}fed_cifar100_test.pkl', num_shards, worker_idx, num_client, train=False, log_prefix='Test ')

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
            use_prefetcher=False,
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
            num_aug_splits= num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=1,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_mem,
            use_multi_epochs_loader=args.use_multi_epochs_loader
        )

        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=False,
            use_prefetcher=False,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
        )
        self.model, self.optimizer, self.lr_scheduler, self.loader_train, self.loader_eval, self.train_loss_fn, self.mixup_fn = model, optimizer, lr_scheduler, loader_train, loader_eval, train_loss_fn, mixup_fn
        print("ConvMLP init finish")

    def init(self):
        return self.model, self.optimizer, self.lr_scheduler, self.loader_train, self.loader_eval, self.train_loss_fn, self.mixup_fn
    
if __name__ == '__main__':
    noise = 'image_blur_'
    d = FedCIAFR100(f'datasets/{noise}fed_cifar100_test.pkl', 4, 0, 4)
    d = FedCIAFR100(f'datasets/{noise}fed_cifar100_test.pkl',4, 1, 4)
    d = FedCIAFR100(f'datasets/{noise}fed_cifar100_test.pkl',4, 1, 4, train=False)