"""
1. remove support for early exit
2. remove support for knowledge distilling
"""

import logging
import math
import os
import parser
import sys
import time
import traceback
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torchnet.meter as tnt

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import distiller
import distiller.apputils as apputils
import distiller.quantization as quantization
import examples.automated_deep_compression as adc
from distiller.data_loggers import *

sys.path.append('/home/CORP.PKUSC.ORG/hatsu3/research/compression/distiller/examples/openpose_compression/pytorch_openpose')
from network import rtpose_shufflenetV2, rtpose_hourglass, rtpose_vgg
from training.datasets.coco import get_loader
from evaluate.coco_eval import run_eval


def hourglass_get_loss(saved_for_loss, heat_temp, heat_weight, vec_temp, vec_weight):
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(size_average=False).cuda()
    batch_size = heat_temp.size(0)

    total_loss = 0

    pred1 = saved_for_loss[0] * vec_weight
    gt1 = vec_temp * vec_weight
    pred2 = saved_for_loss[1] * heat_weight
    gt2 = heat_weight * heat_temp

    loss1 = criterion(pred1, gt1) / (2 * batch_size)
    loss2 = criterion(pred2, gt2) / (2 * batch_size)

    total_loss += loss1
    total_loss += loss2

    saved_for_log['paf'] = loss1.item()
    saved_for_log['heatmap'] = loss2.item()
    saved_for_log['max_ht'] = torch.max(saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data).item()
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data).item()

    return total_loss, saved_for_log


def shufflenetv2_get_loss(saved_for_loss, heat_temp, heat_weight, vec_temp, vec_weight):
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(size_average=True).cuda()
    total_loss = 0

    pred1 = saved_for_loss[0] * vec_weight
    gt1 = vec_temp * vec_weight
    pred2 = saved_for_loss[1] * heat_weight
    gt2 = heat_weight * heat_temp

    loss1 = criterion(pred1, gt1)
    loss2 = criterion(pred2, gt2)

    total_loss += loss1
    total_loss += loss2

    saved_for_log['paf'] = loss1.item()
    saved_for_log['heatmap'] = loss2.item()

    saved_for_log['max_ht'] = torch.max(saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data).item()
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data).item()

    return total_loss, saved_for_log


def vgg19_get_loss(saved_for_loss, heat_temp, heat_weight, vec_temp, vec_weight):
    names = names = ['loss_stage%d_L%d' % (j, k) for j in range(1, 7) for k in range(1, 3)]
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(size_average=True).cuda()
    total_loss = 0

    for j in range(6):
        pred1 = saved_for_loss[2 * j] * vec_weight
        gt1 = vec_temp * vec_weight
        pred2 = saved_for_loss[2 * j + 1] * heat_weight
        gt2 = heat_weight * heat_temp

        loss1 = criterion(pred1, gt1)
        loss2 = criterion(pred2, gt2)

        total_loss += loss1
        total_loss += loss2

        saved_for_log[names[2 * j]] = loss1.item()
        saved_for_log[names[2 * j + 1]] = loss2.item()

    saved_for_log['max_ht'] = torch.max(saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data).item()
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data).item()

    return total_loss, saved_for_log


def create_pose_estimation_model(pretrained, dataset, arch, load_vgg19=None, parallel=True, device_ids=None):
    # noinspection PyGlobalUndefined
    global msglogger

    model = None
    dataset = dataset.lower()

    if dataset == 'coco':
        if arch == 'shufflenetv2':
            model = rtpose_shufflenetV2.Network(width_multiplier=1.0)
            if pretrained:
                msglogger.info('No pretrained ShuffleNetV2 model available. Init randomly.')
        elif arch == 'vgg19':
            model = rtpose_vgg.get_model(trunk='vgg19')
            if pretrained:
                model_dir = Path('./pretrained')
                model_dir.mkdir(exist_ok=True)
                rtpose_vgg.use_vgg(model, model_path, 'vgg19')
            if load_vgg19:
                model.load_state_dict(torch.load(load_vgg19))
        elif arch == 'hourglass':
            model = rtpose_hourglass.hg(num_stacks=8, num_blocks=1, paf_classes=38, ht_classes=19)
            if pretrained:
                msglogger.info('No pretrained Hourglass model available. Init randomly.')
    else:
        raise ValueError('Could not recognize dataset {}'.format(dataset))

    msglogger.info("=> creating a %s%s model with the %s dataset" % (
        'pretrained ' if pretrained else '', arch, dataset))

    if torch.cuda.is_available() and device_ids != -1:
        device = 'cuda'
        if parallel:
            print('Data parallel: device_ids =', device_ids)
            net = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        device = 'cpu'

    return model.to(device)


DATASETS_NAMES = ['coco']


def load_data_(dataset, arch, json_path, data_dir, mask_dir, batch_size, workers):
    if dataset not in DATASETS_NAMES:
        raise ValueError('load_data does not support dataset %s" % dataset')

    input_size = {
        'shufflenetv2': 368,
        'vgg19': 368,
        'hourglass': 256,
    }[arch]

    input_shape = (1, 3, input_size, input_size)

    params_transform = {
        'shufflenetv2': {
            'mode': 5,
            'scale_min': 0.5,
            'scale_max': 1.1,
            'scale_prob': 1,
            'target_dist': 0.6,
            'max_rotate_degree': 40,
            'center_perterb_max': 40,
            'flip_prob': 0.5,
            'np': 56,
            'sigma': 7.0,
        },
        'hourglass': {
            'mode': 5,
            'scale_min': 0.5,
            'scale_max': 1.1,
            'scale_prob': 1,
            'target_dist': 0.6,
            'max_rotate_degree': 40,
            'center_perterb_max': 40,
            'flip_prob': 0.5,
            'np': 56,
            'sigma': 4.416,
            'limb_width': 1.289,
        },
        'vgg19': {
            'mode': 5,
            'scale_min': 0.5,
            'scale_max': 1.1,
            'scale_prob': 1,
            'target_dist': 0.6,
            'max_rotate_degree': 40,
            'center_perterb_max': 40,
            'flip_prob': 0.5,
            'np': 56,
            'sigma': 7.0,
            'limb_width': 1.,
        }
    }[arch]

    if arch == 'shufflenetv2':
        train_loader = get_loader(json_path, data_dir, mask_dir, input_size, 8,
                                  preprocess='rtpose', batch_size=batch_size, params_transform=params_transform,
                                  shuffle=True, training=True, num_workers=workers)

        valid_loader = get_loader(json_path, data_dir, mask_dir, input_size, 8,
                                  preprocess='rtpose', batch_size=batch_size, params_transform=params_transform,
                                  shuffle=False, training=False, num_workers=workers)
    elif arch == 'hourglass':
        train_loader = get_loader(json_path, data_dir, mask_dir, input_size, 4,
                                  preprocess='rtpose', batch_size=batch_size, params_transform=params_transform,
                                  shuffle=True, training=True, num_workers=workers)

        valid_loader = get_loader(json_path, data_dir, mask_dir, input_size, 4,
                                  preprocess='rtpose', batch_size=batch_size, params_transform=params_transform,
                                  shuffle=False, training=False, num_workers=workers)
    elif arch == 'vgg19':
        train_loader = get_loader(json_path, data_dir, mask_dir, input_size, 8,
                                  preprocess='vgg', batch_size=batch_size, params_transform=params_transform,
                                  shuffle=True, training=True, num_workers=workers)

        valid_loader = get_loader(json_path, data_dir, mask_dir, input_size, 8,
                                  preprocess='vgg', batch_size=batch_size, params_transform=params_transform,
                                  shuffle=False, training=False, num_workers=workers)
    else:
        raise ValueError

    return train_loader, valid_loader, valid_loader, input_shape


def load_data(args):
    return load_data_(args.dataset, args.arch, args.json_path, os.path.expanduser(args.data),
                      args.mask_dir, args.batch_size, args.workers)


def train(train_loader, model, criterion, optimizer, epoch, compression_scheduler, loggers, args):
    """Training loop for one epoch."""

    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()
    losses = tnt.AverageValueMeter()

    meter_dict = {'paf': tnt.AverageValueMeter(), 'heatmap': tnt.AverageValueMeter(),
                  'max_ht': tnt.AverageValueMeter(), 'min_ht': tnt.AverageValueMeter(),
                  'max_paf': tnt.AverageValueMeter(), 'min_paf': tnt.AverageValueMeter()}

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

    model.train()
    end = time.time()

    for train_step, (inputs, heatmap_target, heat_mask, paf_target, paf_mask) in enumerate(train_loader):
        data_time.add(time.time() - end)

        inputs = inputs.to(args.device)
        heatmap_target = heatmap_target.to(args.device)
        heat_mask = heat_mask.to(args.device)
        paf_target = paf_target.to(args.device)
        paf_mask = paf_mask.to(args.device)

        # Execute the forward phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

        _, saved_for_loss = model(inputs)
        # criterion: get_loss
        total_loss, saved_for_log = criterion(saved_for_loss, heatmap_target, heat_mask, paf_target, paf_mask)

        for name, _ in meter_dict.items():
            meter_dict[name].add(saved_for_log[name], inputs.size(0))
        losses.add(total_loss, inputs.size(0))

        # TODO: remove?
        if compression_scheduler:
            # Before running the backward phase, we allow the scheduler to modify the loss
            # (e.g. add regularization loss)
            agg_loss = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, total_loss,
                                                                  optimizer=optimizer, return_loss_components=True)
            loss = agg_loss.overall_loss
            losses['overall_loss'].add(loss.item())

            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                losses[lc.name].add(lc.value.item())
        else:
            losses['overall_loss'].add(total_loss.item())

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        if compression_scheduler:
            compression_scheduler.before_parameter_optimization(epoch, train_step, steps_per_epoch, optimizer)
        optimizer.step()
        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)

        batch_time.add(time.time() - end)
        steps_completed = (train_step + 1)

        if steps_completed % args.print_freq == 0:
            stats_dict = OrderedDict({
                'loss': losses.mean,
                'LR': optimizer.param_groups[0]['lr'],
                'Time': batch_time.mean,
            })
            stats = ('Performance/Training/', stats_dict)

            params = model.named_parameters() if args.log_params_histograms else None
            distiller.log_training_progress(stats, params, epoch, steps_completed,
                                            steps_per_epoch, args.print_freq, loggers)
        end = time.time()

    return losses.mean


def validate(val_loader, model, criterion, loggers, args, epoch=-1):
    """Model validation"""
    if epoch > -1:
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, loggers, args, epoch)


def _validate(data_loader, model, criterion, loggers, args, epoch=-1):
    """Execute the validation/test loop."""

    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()
    losses = tnt.AverageValueMeter()

    meter_dict = {'paf': tnt.AverageValueMeter(), 'heatmap': tnt.AverageValueMeter(),
                  'max_ht': tnt.AverageValueMeter(), 'min_ht': tnt.AverageValueMeter(),
                  'max_paf': tnt.AverageValueMeter(), 'min_paf': tnt.AverageValueMeter()}

    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    total_steps = total_samples / batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    model.eval()  # TODO: model.train() in original repo
    end = time.time()

    # model = torch.nn.DataParallel(model, device_ids=args.gpus)
    # run_eval(image_dir=args.data, anno_dir=args.anno_dir, vis_dir=args.vis_dir,
    #          image_list_txt=args.image_list_txt,
    #          model=model, preprocess='vgg' if args.arch == 'vgg19' else 'rtpose')

    for validation_step, (inputs, heatmap_target, heat_mask, paf_target, paf_mask) in enumerate(data_loader):
        with torch.no_grad():
            data_time.add(time.time() - end)

            inputs = inputs.to(args.device)
            heatmap_target = heatmap_target.to(args.device)
            heat_mask = heat_mask.to(args.device)
            paf_target = paf_target.to(args.device)
            paf_mask = paf_mask.to(args.device)

            _, saved_for_loss = model(inputs)
            total_loss, saved_for_log = criterion(saved_for_loss, heatmap_target, heat_mask, paf_target, paf_mask)

            losses.add(total_loss.item(), inputs.size(0))

            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = (validation_step + 1)
            if steps_completed % args.print_freq == 0:
                stats = ('', OrderedDict([('Loss', losses.mean), ]))
                distiller.log_training_progress(stats, None, epoch, steps_completed,
                                                total_steps, args.print_freq, loggers)

    msglogger.info('==> Loss: %.6f\n', losses.mean)

    # TODO: refactor me
    with open('/home/CORP.PKUSC.ORG/hatsu3/research/compression/distiller/examples/openpose_compression/notebooks/results.txt', 'w') as f:
        f.write('%.6f' % losses.mean)

    return losses.mean


def test(test_loader, model, criterion, loggers, activations_collectors, args):
    """Model Test"""
    msglogger.info('--- test ---------------------')
    if activations_collectors is None:
        activations_collectors = create_activation_stats_collectors(model, None)

    with collectors_context(activations_collectors["test"]) as collectors:
        loss = _validate(test_loader, model, criterion, loggers, args)
        distiller.log_activation_statsitics(-1, "test", loggers, collector=collectors['sparsity'])
        save_collectors_data(collectors, msglogger.logdir)

    return loss


# Logger handle
msglogger = None


def main():
    script_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
    global msglogger

    # Parse arguments
    args = parser.get_parser().parse_args()
    if args.epochs is None:
        args.epochs = 200

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir)

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    apputils.log_execution_env_state(
        filter(None, [args.compress, args.qe_stats_file]),  # remove both None and empty strings
        msglogger.logdir, gitroot=module_path)
    msglogger.debug("Distiller: %s", distiller.__version__)

    if args.evaluate:
        args.deterministic = True
    if args.deterministic:
        distiller.set_deterministic(args.seed)  # For experiment reproducability
    else:
        if args.seed is not None:
            distiller.set_seed(args.seed)
        # Turn on CUDNN benchmark mode for best performance. This is usually "safe" for image
        # classification models, as the input sizes don't change during the run
        # See here: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
        cudnn.benchmark = True

    start_epoch = 0
    ending_epoch = args.epochs
    perf_scores_history = []

    if args.cpu or not torch.cuda.is_available():
        # Set GPU index to -1 if using CPU
        args.device = 'cpu'
        args.gpus = -1
    else:
        args.device = 'cuda'
        if args.gpus is not None:
            try:
                args.gpus = [int(s) for s in args.gpus.split(',')]
            except ValueError:
                raise ValueError('ERROR: Argument --gpus must be a comma-separated list of integers only')
            available_gpus = torch.cuda.device_count()
            for dev_id in args.gpus:
                if dev_id >= available_gpus:
                    raise ValueError('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                     .format(dev_id, available_gpus))
            # Set default device in case the first one on the list != 0
            torch.cuda.set_device(args.gpus[0])

    # Infer the dataset from the model name
    # TODO
    args.dataset = 'coco'
    # args.num_classes = 21  # wc -l ~/data/VOC2012/voc-model-labels.txt

    if args.load_vgg19 and args.arch != 'vgg19':
        raise ValueError('``load_vgg19`` should be set only when vgg19 is used')

    model = create_pose_estimation_model(
        args.pretrained, args.dataset, args.arch,
        load_vgg19=args.load_vgg19,
        parallel=not args.load_serialized,
        device_ids=args.gpus)
    compression_scheduler = None
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    tflogger = TensorBoardLogger(msglogger.logdir)
    pylogger = PythonLogger(msglogger)

    # <editor-fold desc=">>> Load Model">

    # We can optionally resume from a checkpoint
    optimizer = None
    if args.resumed_checkpoint_path:
        model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(
            model, args.resumed_checkpoint_path, model_device=args.device)
    elif args.load_model_path:
        model = apputils.load_lean_checkpoint(model, args.load_model_path, model_device=args.device)

    if args.reset_optimizer:
        start_epoch = 0
        if optimizer is not None:
            optimizer = None
            msglogger.info('\nreset_optimizer flag set: Overriding resumed optimizer and resetting epoch count to 0')
    # </editor-fold>

    # Define loss function (criterion)
    # get_loss(saved_for_loss, heat_temp, heat_weight,vec_temp, vec_weight)
    criterion = {
        'shufflenetv2': shufflenetv2_get_loss,
        'vgg19': vgg19_get_loss,
        'hourglass': hourglass_get_loss,
    }[args.arch]

    if optimizer is None:
        trainable_vars = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        msglogger.info('Optimizer Type: %s', type(optimizer))
        msglogger.info('Optimizer Args: %s', optimizer.defaults)

    # TODO: load lr_scheduler
    lr_scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001,
        threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

    if args.AMC:
        return automated_deep_compression(model, criterion, optimizer, pylogger, args)
    if args.greedy:
        return greedy(model, criterion, optimizer, pylogger, args)

    # This sample application can be invoked to produce various summary reports.
    if args.summary:
        for summary in args.summary:
            distiller.model_summary(model, summary, args.dataset)
        return

    if args.export_onnx is not None:
        return distiller.export_img_classifier_to_onnx(
            model, os.path.join(msglogger.logdir, args.export_onnx),
            args.dataset, add_softmax=True, verbose=False)

    if args.qe_calibration:
        return acts_quant_stats_collection(model, criterion, pylogger, args)

    if args.activation_histograms:
        return acts_histogram_collection(model, criterion, pylogger, args)

    print('Building activations_collectors...')
    activations_collectors = create_activation_stats_collectors(model, *args.activation_stats)

    # Load the datasets: the dataset to load is inferred from the model name passed
    # in args.arch.  The default dataset is ImageNet, but if args.arch contains the
    # substring "_cifar", then cifar10 is used.
    print('Loading data...')
    train_loader, val_loader, test_loader, _ = load_data(args)
    msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    if args.sensitivity is not None:
        sensitivities = np.arange(args.sensitivity_range[0], args.sensitivity_range[1], args.sensitivity_range[2])
        return sensitivity_analysis(model, criterion, test_loader, pylogger, args, sensitivities)

    if args.evaluate:
        return evaluate_model(model, criterion, test_loader, pylogger, activations_collectors, args,
                              compression_scheduler)

    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.file_config(
            model, optimizer, args.compress, compression_scheduler,
            (start_epoch - 1) if args.resumed_checkpoint_path else None
        )
        # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
        model.to(args.device)
    elif compression_scheduler is None:
        compression_scheduler = distiller.CompressionScheduler(model)

    if args.thinnify:
        # zeros_mask_dict = distiller.create_model_masks_dict(model)
        assert args.resumed_checkpoint_path is not None, \
            "You must use --resume-from to provide a checkpoint file to thinnify"
        distiller.remove_filters(model, compression_scheduler.zeros_mask_dict, args.arch, args.dataset, optimizer=None)
        apputils.save_checkpoint(0, args.arch, model, optimizer=None, scheduler=compression_scheduler,
                                 name="{}_thinned".format(args.resumed_checkpoint_path.replace(".pth.tar", "")),
                                 dir=msglogger.logdir)
        print("Note: your model may have collapsed to random inference, so you may want to fine-tune")
        return

    if start_epoch >= ending_epoch:
        msglogger.error(
            'epoch count is too low, starting epoch is {} but total epochs set to {}'.format(
                start_epoch, ending_epoch))
        raise ValueError('Epochs parameter is too low. Nothing to do.')

    for epoch in range(start_epoch, ending_epoch):
        # This is the main training loop.
        msglogger.info('\n')
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch, metrics=(total_loss if (epoch != start_epoch) else 10 ** 6))

        # Train for one epoch
        with collectors_context(activations_collectors["train"]) as collectors:
            train(train_loader, model, criterion, optimizer, epoch, compression_scheduler,
                  loggers=[tflogger, pylogger], args=args)
            distiller.log_weights_sparsity(model, epoch, loggers=[tflogger, pylogger])
            distiller.log_activation_statsitics(epoch, "train", loggers=[tflogger],
                                                collector=collectors["sparsity"])
            if args.masks_sparsity:
                msglogger.info(distiller.masks_sparsity_tbl_summary(model, compression_scheduler))

        # evaluate on validation set
        with collectors_context(activations_collectors["valid"]) as collectors:
            loss = validate(val_loader, model, criterion, [pylogger], args, epoch)
            distiller.log_activation_statsitics(epoch, "valid", loggers=[tflogger], collector=collectors["sparsity"])
            save_collectors_data(collectors, msglogger.logdir)

        lr_scheduler.step(loss)

        stats = ('Performance/Validation/', OrderedDict([('Loss', loss)]))
        distiller.log_training_progress(stats, None, epoch, steps_completed=0, total_steps=1, log_freq=1,
                                        loggers=[tflogger])

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)

        # Update the list of top scores achieved so far, and save the checkpoint
        update_training_scores_history(perf_scores_history, model, loss, epoch, args.num_best_scores)
        is_best = epoch == perf_scores_history[0].epoch
        checkpoint_extras = {'current_loss': loss,
                             'best_loss': perf_scores_history[0].loss,
                             'best_epoch': perf_scores_history[0].epoch}
        apputils.save_checkpoint(epoch, args.arch, model, optimizer=optimizer, scheduler=compression_scheduler,
                                 extras=checkpoint_extras, is_best=is_best, name=args.name, dir=msglogger.logdir)

    # Finally run results on the test set
    test(test_loader, model, criterion, [pylogger], activations_collectors, args=args)


def update_training_scores_history(perf_scores_history, model, loss, epoch, num_best_scores):
    """ Update the list of top training scores achieved so far, and log the best scores so far"""

    model_sparsity, _, params_nnz_cnt = distiller.model_params_stats(model)
    perf_scores_history.append(distiller.MutableNamedTuple({'params_nnz_cnt': -params_nnz_cnt,
                                                            'sparsity': model_sparsity,
                                                            'loss': loss,
                                                            'epoch': epoch}))

    # Keep perf_scores_history sorted from best to worst
    # Sort by sparsity as main sort key, then sort by top1, top5 and epoch

    def key(entry):
        return (
            entry['params_nnz_cnt'],
            -entry['loss'],
            entry['epoch'],
        )

    perf_scores_history.sort(key=key, reverse=True)
    for score in perf_scores_history[:num_best_scores]:
        msglogger.info(
            '==> Best [Loss: %.3f   Sparsity:%.2f   Params: %d on epoch: %d]',
            score.loss, score.sparsity, -score.params_nnz_cnt, score.epoch)


def evaluate_model(model, criterion, test_loader, loggers, activations_collectors, args, scheduler=None):
    # This sample application can be invoked to evaluate the accuracy of your model on
    # the test dataset.
    # You can optionally quantize the model to 8-bit integer before evaluation.
    # For example:
    # python3 compress_classifier.py --arch resnet20_cifar
    # ../data.cifar10 -p=50 --resume-from=checkpoint.pth.tar --evaluate

    if not isinstance(loggers, list):
        loggers = [loggers]

    if args.quantize_eval:
        model.cpu()
        quantizer = quantization.PostTrainLinearQuantizer.from_args(model, args)
        quantizer.prepare_model()
        model.to(args.device)

    loss = test(test_loader, model, criterion, loggers, activations_collectors, args=args)

    if args.quantize_eval:
        checkpoint_name = 'quantized'
        apputils.save_checkpoint(0, args.arch, model, optimizer=None, scheduler=scheduler,
                                 name='_'.join([args.name, checkpoint_name]) if args.name else checkpoint_name,
                                 dir=msglogger.logdir, extras={'quantized_loss': loss})


def sensitivity_analysis(model, criterion, data_loader, loggers, args, sparsities):
    # This sample application can be invoked to execute Sensitivity Analysis on your
    # model.  The ouptut is saved to CSV and PNG.
    msglogger.info("Running sensitivity tests")
    if not isinstance(loggers, list):
        loggers = [loggers]
    test_fnc = partial(test, test_loader=data_loader, criterion=criterion,
                       loggers=loggers, args=args,
                       activations_collectors=create_activation_stats_collectors(model))
    which_params = [param_name for param_name, _ in model.named_parameters()]
    sensitivity = distiller.perform_sensitivity_analysis(model,
                                                         net_params=which_params,
                                                         sparsities=sparsities,
                                                         test_func=test_fnc,
                                                         group=args.sensitivity)
    distiller.sensitivities_to_png(sensitivity, 'sensitivity.png')
    distiller.sensitivities_to_csv(sensitivity, 'sensitivity.csv')


def automated_deep_compression(model, criterion, optimizer, loggers, args):
    train_loader, val_loader, test_loader, _ = load_data(args)

    args.display_confusion = True
    validate_fn = partial(test, test_loader=test_loader, criterion=criterion,
                          loggers=loggers, args=args, activations_collectors=None)
    train_fn = partial(train, train_loader=train_loader, criterion=criterion,
                       loggers=loggers, args=args)

    save_checkpoint_fn = partial(apputils.save_checkpoint, arch=args.arch, dir=msglogger.logdir)
    optimizer_data = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay}
    adc.do_adc(model, args, optimizer_data, validate_fn, save_checkpoint_fn, train_fn)


def greedy(model, criterion, optimizer, loggers, args):
    train_loader, val_loader, test_loader, _ = load_data(args)

    test_fn = partial(test, test_loader=test_loader, criterion=criterion,
                      loggers=loggers, args=args, activations_collectors=None)
    train_fn = partial(train, train_loader=train_loader, criterion=criterion, args=args)
    assert args.greedy_target_density is not None
    distiller.pruning.greedy_filter_pruning.greedy_pruner(model, args,
                                                          args.greedy_target_density,
                                                          args.greedy_pruning_step,
                                                          test_fn, train_fn)


def acts_quant_stats_collection(model, criterion, loggers, args):
    msglogger.info('Collecting quantization calibration stats based on {:.1%} of test dataset'
                   .format(args.qe_calibration))
    model = distiller.utils.make_non_parallel_copy(model)
    args.effective_test_size = args.qe_calibration
    train_loader, val_loader, test_loader, _ = load_data(args)
    test_fn = partial(test, test_loader=test_loader, criterion=criterion,
                      loggers=loggers, args=args, activations_collectors=None)
    collect_quant_stats(model, test_fn, save_dir=msglogger.logdir,
                        classes=None, inplace_runtime_check=True, disable_inplace_attrs=True)


def acts_histogram_collection(model, criterion, loggers, args):
    msglogger.info('Collecting activation histograms based on {:.1%} of test dataset'
                   .format(args.activation_histograms))
    model = distiller.utils.make_non_parallel_copy(model)
    args.effective_test_size = args.activation_histograms
    train_loader, val_loader, test_loader, _ = load_data(args, fixed_subset=True)
    test_fn = partial(test, test_loader=test_loader, criterion=criterion,
                      loggers=loggers, args=args, activations_collectors=None)
    collect_histograms(model, test_fn, save_dir=msglogger.logdir,
                       classes=None, nbins=2048, save_hist_imgs=True)


class missingdict(dict):
    """This is a little trick to prevent KeyError"""

    def __missing__(self, key):
        return None  # note, does *not* set self[key] - we don't want defaultdict's behavior


def create_activation_stats_collectors(model, *phases):
    """Create objects that collect activation statistics.

    This is a utility function that creates two collectors:
    1. Fine-grade sparsity levels of the activations
    2. L1-magnitude of each of the activation channels

    Args:
        model - the model on which we want to collect statistics
        phases - the statistics collection phases: train, valid, and/or test

    WARNING! Enabling activation statsitics collection will significantly slow down training!
    """
    genCollectors = lambda: missingdict({
        "sparsity": SummaryActivationStatsCollector(model, "sparsity",
                                                    lambda t: 100 * distiller.utils.sparsity(t)),
        "l1_channels": SummaryActivationStatsCollector(model, "l1_channels",
                                                       distiller.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         distiller.utils.activation_channels_apoz),
        "mean_channels": SummaryActivationStatsCollector(model, "mean_channels",
                                                         distiller.utils.activation_channels_means),
        "records": RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])
    })

    return {k: (genCollectors() if k in phases else missingdict())
            for k in ('train', 'valid', 'test')}


def save_collectors_data(collectors, directory):
    """Utility function that saves all activation statistics to disk.

    File type and format of contents are collector-specific.
    """
    for name, collector in collectors.items():
        msglogger.info('Saving data for collector {}...'.format(name))
        file_path = collector.save(os.path.join(directory, name))
        msglogger.info("Saved to {}".format(file_path))


def check_pytorch_version():
    from pkg_resources import parse_version
    if parse_version(torch.__version__) < parse_version('1.0.1'):
        print("\nNOTICE:")
        print("The Distiller \'master\' branch now requires at least PyTorch version 1.0.1 due to "
              "PyTorch API changes which are not backward-compatible.\n"
              "Please install PyTorch 1.0.1 or its derivative.\n"
              "If you are using a virtual environment, do not forget to update it:\n"
              "  1. Deactivate the old environment\n"
              "  2. Install the new environment\n"
              "  3. Activate the new environment")
        exit(1)


if __name__ == '__main__':
    try:
        check_pytorch_version()
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
