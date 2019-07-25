import sys
import itertools

import torch
from torch.utils.data import ConcatDataset

from pytorch_ssd.vision.datasets.open_images import OpenImagesDataset
from pytorch_ssd.vision.datasets.voc_dataset import VOCDataset
from pytorch_ssd.vision.nn.multibox_loss import MultiboxLoss
from pytorch_ssd.vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from pytorch_ssd.vision.utils.misc import store_labels


def width_mult_from_arch(arch):
    return float(arch.split('-')[-1])


def detection_dataset_str_from_arch(arch): ...


def detection_num_classes(dataset):
    return 80


def get_scheduler(scheduler):
    if scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=last_epoch)
    elif scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)
