"""
1. remove support for early exit
2. remove support for knowledge distilling
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import itertools
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
import torchnet.meter as tnt

sys.path.append('./pytorch_ssd')

from pytorch_ssd.vision.datasets.voc_dataset import VOCDataset
from pytorch_ssd.vision.nn.multibox_loss import MultiboxLoss
from pytorch_ssd.vision.ssd.config import mobilenetv1_ssd_config
from pytorch_ssd.vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from pytorch_ssd.vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from pytorch_ssd.vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite_predictor
from pytorch_ssd.vision.ssd.ssd import MatchPrior
from pytorch_ssd.vision.utils import freeze_net_layers, box_utils, measurements

import distiller
import distiller.apputils as apputils
import distiller.quantization as quantization
import examples.automated_deep_compression as adc
from distiller.apputils.data_loaders import get_data_loaders
from distiller.data_loggers import *

from tqdm.auto import tqdm


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index] = {}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


def create_detection_model(pretrained, dataset, arch,
                           lr=None, base_net_lr=None, extra_layers_lr=None,
                           freeze_base_net=False, freeze_net=False,
                           parallel=True, device_ids=None):
    # noinspection PyGlobalUndefined
    global msglogger

    # if pretrained:
    #     raise ValueError('Pretrained model not available')

    net = None
    dataset = dataset.lower()
    # TODO: onnx_compatible?
    # arch not used
    if dataset == 'voc':
        num_classes = 21
        width_mult = 1.0
        net = create_mobilenetv2_ssd_lite(
            num_classes, width_mult=width_mult, use_batch_norm=True,
            onnx_compatible=False, is_test=False)

        if pretrained:
            net.load('pretrained/mb2-ssd-lite-mp-0_686.pth')

        base_net_lr = base_net_lr if base_net_lr is not None else lr
        extra_layers_lr = extra_layers_lr if extra_layers_lr is not None else lr

        if freeze_base_net:
            msglogger.info("=> Freeze base net.")
            freeze_net_layers(net.base_net)
            # params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
            #                          net.regression_headers.parameters(), net.classification_headers.parameters())
            params = [
                {'params': itertools.chain(
                    net.source_layer_add_ons.parameters(),
                    net.extras.parameters()
                ), 'lr': extra_layers_lr},
                {'params': itertools.chain(
                    net.regression_headers.parameters(),
                    net.classification_headers.parameters()
                )}
            ]
        elif freeze_net:
            freeze_net_layers(net.base_net)
            freeze_net_layers(net.source_layer_add_ons)
            freeze_net_layers(net.extras)
            params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
            msglogger.info("=> Freeze all the layers except prediction heads.")
        else:
            params = [
                {'params': net.base_net.parameters(), 'lr': base_net_lr},
                {'params': itertools.chain(
                    net.source_layer_add_ons.parameters(),
                    net.extras.parameters()
                ), 'lr': extra_layers_lr},
                {'params': itertools.chain(
                    net.regression_headers.parameters(),
                    net.classification_headers.parameters()
                )}
            ]
    else:
        raise ValueError('Could not recognize dataset {}'.format(dataset))

    msglogger.info("=> creating a %s%s model with the %s dataset" % (
        'pretrained ' if pretrained else '', arch, dataset))
    if torch.cuda.is_available() and device_ids != -1:
        device = 'cuda'
        if (arch.startswith('alexnet') or arch.startswith('vgg')) and parallel:
            net.features = torch.nn.DataParallel(net.features, device_ids=device_ids)
        elif parallel:
            net = torch.nn.DataParallel(net, device_ids=device_ids)
    else:
        device = 'cpu'

    return net.to(device), params


DATASETS_NAMES = ['voc']


def __dataset_factory(dataset): return {'voc': voc_get_datasets, }[dataset]


def voc_get_datasets(data_dir, config=mobilenetv1_ssd_config):
    """
    Load the VOC dataset.
    """
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    train_dataset = VOCDataset(data_dir, transform=train_transform, target_transform=target_transform)
    test_dataset = VOCDataset(data_dir, transform=test_transform, target_transform=target_transform, is_test=True)

    return train_dataset, test_dataset


def load_data_(dataset, data_dir, batch_size, workers, validation_split=0.1, deterministic=False,
               effective_train_size=1., effective_valid_size=1., effective_test_size=1.,
               fixed_subset=False, sequential=False):
    if dataset not in DATASETS_NAMES:
        raise ValueError('load_data does not support dataset %s" % dataset')
    datasets_fn = __dataset_factory(dataset)
    return get_data_loaders(datasets_fn, data_dir, batch_size, workers,
                            validation_split=validation_split,
                            deterministic=deterministic,
                            effective_train_size=effective_train_size,
                            effective_valid_size=effective_valid_size,
                            effective_test_size=effective_test_size,
                            fixed_subset=fixed_subset,
                            sequential=sequential)


def load_data(args, fixed_subset=False):
    return load_data_(args.dataset, os.path.expanduser(args.data), args.batch_size,
                      args.workers, args.validation_split, args.deterministic,
                      args.effective_train_size, args.effective_valid_size, args.effective_test_size,
                      fixed_subset)


def train(train_loader, model, criterion, optimizer, epoch,
          compression_scheduler, loggers, args):
    """Training loop for one epoch."""
    losses = {
        'overall_loss': tnt.AverageValueMeter(),
        'total_loss': tnt.AverageValueMeter(),
        'regression_loss': tnt.AverageValueMeter(),
        'classification_loss': tnt.AverageValueMeter(),
    }
    # classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

    model.train()
    acc_stats = []
    end = time.time()

    for train_step, (inputs, boxes, labels) in enumerate(train_loader):
        data_time.add(time.time() - end)
        inputs = inputs.to(args.device)
        boxes = boxes.to(args.device)
        labels = labels.to(args.device)

        # Execute the forward phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

        confidence, locations = model(inputs)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        total_loss = regression_loss + classification_loss

        # classerr.add(output.data, target)
        # acc_stats.append([classerr.value(1), classerr.value(5)])
        acc_stats.append([total_loss.item(), regression_loss.item(), classification_loss.item()])
        losses['total_loss'].add(total_loss.item())
        losses['regression_loss'].add(regression_loss.item())
        losses['classification_loss'].add(classification_loss.item())

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

        # measure elapsed time
        batch_time.add(time.time() - end)
        steps_completed = (train_step + 1)

        if steps_completed % args.print_freq == 0:
            # Log some statistics
            # errs = OrderedDict({
            #     'Top1': classerr.value(1),
            #     'Top5': classerr.value(5)
            # })

            stats_dict = OrderedDict()
            for loss_name, meter in losses.items():
                stats_dict[loss_name] = meter.mean
            # stats_dict.update(errs)
            stats_dict['LR'] = optimizer.param_groups[0]['lr']
            stats_dict['Time'] = batch_time.mean
            stats = ('Performance/Training/', stats_dict)

            params = model.named_parameters() if args.log_params_histograms else None
            distiller.log_training_progress(stats, params, epoch, steps_completed,
                                            steps_per_epoch, args.print_freq, loggers)
        end = time.time()
    return acc_stats


def validate(val_loader, model, criterion, loggers, args, epoch=-1):
    """Model validation"""
    if epoch > -1:
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, loggers, args, epoch)


def measure_map(model, args, eval_path, nms_method='hard', label_file='~/data/VOC2012/voc-model-labels.txt'):
    if args.dataset != 'voc':
        raise ValueError('only VOC dataset is supported in measure_map')

    data_dir = os.path.expanduser(args.data)
    dataset = VOCDataset(data_dir, is_test=True)
    label_file = Path(label_file).expanduser()
    class_names = [name.strip() for name in open(label_file).readlines()]
    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    predictor = create_mobilenetv2_ssd_lite_predictor(model, nms_method=nms_method, device=args.device)

    results = []
    for i in tqdm(range(len(dataset))):
        image = dataset.get_image(i)
        boxes, labels, probs = predictor.predict(image)
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))

    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )

    aps = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            args.iou_threshold,
            args.use_2007_metric
        )
        aps.append(ap)
        print(f"{class_name}: {ap}")

    print(f"\nAverage Precision Across All Classes:{sum(aps) / len(aps)}")


def _validate(data_loader, model, criterion, loggers, args, epoch=-1):
    """Execute the validation/test loop."""
    losses = {
        'total_loss': tnt.AverageValueMeter(),
        'regression_loss': tnt.AverageValueMeter(),
        'classification_loss': tnt.AverageValueMeter(),
    }
    # classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))

    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    if args.display_confusion:
        confusion = tnt.ConfusionMeter(args.num_classes)
    total_steps = total_samples / batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    model.eval()

    end = time.time()
    for validation_step, (inputs, boxes, labels) in enumerate(data_loader):
        with torch.no_grad():
            inputs = inputs.to(args.device)
            boxes = boxes.to(args.device)
            labels = labels.to(args.device)

            confidence, locations = model(inputs)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            total_loss = regression_loss + classification_loss

            losses['total_loss'].add(total_loss.item())
            losses['regression_loss'].add(regression_loss.item())
            losses['classification_loss'].add(classification_loss.item())

            # classerr.add(confidence.data, labels)
            if args.display_confusion:
                # noinspection PyUnboundLocalVariable
                confusion.add(output.data, target)

            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = (validation_step + 1)
            if steps_completed % args.print_freq == 0:
                stats = ('',
                         OrderedDict([('Total Loss', losses['total_loss'].mean),
                                      ('Regression Loss', losses['regression_loss'].mean),
                                      ('Classification Loss', losses['classification_loss'].mean)]))
                # ('Top1', classerr.value(1)),
                # ('Top5', classerr.value(5))]))

                distiller.log_training_progress(stats, None, epoch, steps_completed,
                                                total_steps, args.print_freq, loggers)

    # msglogger.info(
    #     '==> Top1: %.3f    Top5: %.3f    Total Loss: %.3f    Classification Loss: %.3f    Regression Loss: %.3f\n',
    #     classerr.value()[0], classerr.value()[1], losses['total_loss'].mean,
    #     losses['classification_loss'].mean, losses['regression_loss'].mean)

    msglogger.info(
        '==> Total Loss: %.3f    Classification Loss: %.3f    Regression Loss: %.3f\n',
        losses['total_loss'].mean, losses['classification_loss'].mean, losses['regression_loss'].mean)

    if args.display_confusion:
        msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))
    # return (classerr.value(1), classerr.value(5),
    #         losses['total_loss'].mean, losses['classification_loss'].mean, losses['regression_loss'].mean)

    return losses['total_loss'].mean, losses['classification_loss'].mean, losses['regression_loss'].mean


def test(test_loader, model, criterion, loggers, activations_collectors, args):
    """Model Test"""
    msglogger.info('--- test ---------------------')
    if activations_collectors is None:
        activations_collectors = create_activation_stats_collectors(model, None)
    with collectors_context(activations_collectors["test"]) as collectors:
        # top1, top5, total_loss, classification_loss, regression_loss = _validate(
        #     test_loader, model, criterion, loggers, args)
        total_loss, classification_loss, regression_loss = _validate(
            test_loader, model, criterion, loggers, args)
        distiller.log_activation_statsitics(-1, "test", loggers, collector=collectors['sparsity'])
        save_collectors_data(collectors, msglogger.logdir)
    from pathlib import Path
    eval_path = Path(msglogger.logdir) / 'ap'
    eval_path.mkdir(exist_ok=True, parents=True)
    if args.calculate_map:
        measure_map(model, args, eval_path=str(eval_path))
    return total_loss, classification_loss, regression_loss


# Logger handle
msglogger = None


def main():
    script_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
    global msglogger

    # Parse arguments
    args = parser.get_parser().parse_args()
    if args.epochs is None:
        args.epochs = 90  # TODO: default training epochs

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
    args.dataset = 'voc'
    args.num_classes = 21  # wc -l ~/data/VOC2012/voc-model-labels.txt
    # args.dataset = distiller.apputils.classification_dataset_str_from_arch(args.arch)
    # args.num_classes = distiller.apputils.classification_num_classes(args.dataset)

    # Create the model
    model, params = create_detection_model(args.pretrained, args.dataset, args.arch, freeze_base_net=False,
                                           freeze_net=False, parallel=not args.load_serialized, device_ids=args.gpus)
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
    elif args.load_base_net:
        msglogger.info("=> loading base net %s", args.load_base_net)
        model.init_from_base_net(args.load_base_net)

    if args.reset_optimizer:
        start_epoch = 0
        if optimizer is not None:
            optimizer = None
            msglogger.info('\nreset_optimizer flag set: Overriding resumed optimizer and resetting epoch count to 0')
    # </editor-fold>

    # Define loss function (criterion)
    config = mobilenetv1_ssd_config  # TODO: get_config
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=args.device)

    if optimizer is None:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        msglogger.info('Optimizer Type: %s', type(optimizer))
        msglogger.info('Optimizer Args: %s', optimizer.defaults)

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
            # top1, top5, total_loss, classification_loss, regression_loss = validate(
            #     val_loader, model, criterion, [pylogger], args, epoch)
            total_loss, classification_loss, regression_loss = validate(
                val_loader, model, criterion, [pylogger], args, epoch)
            distiller.log_activation_statsitics(epoch, "valid", loggers=[tflogger], collector=collectors["sparsity"])
            save_collectors_data(collectors, msglogger.logdir)

        stats = ('Performance/Validation/',
                 OrderedDict([('Total Loss', total_loss), ('Classification Loss', classification_loss),
                              ('Regression Loss', regression_loss)]))
        distiller.log_training_progress(stats, None, epoch, steps_completed=0, total_steps=1, log_freq=1,
                                        loggers=[tflogger])

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)

        # Update the list of top scores achieved so far, and save the checkpoint
        update_training_scores_history(perf_scores_history, model, total_loss, classification_loss, regression_loss,
                                       epoch, args.num_best_scores)
        is_best = epoch == perf_scores_history[0].epoch
        checkpoint_extras = {'current_total_loss': total_loss,
                             'best_total_loss': perf_scores_history[0].total_loss,
                             'best_epoch': perf_scores_history[0].epoch}
        apputils.save_checkpoint(epoch, args.arch, model, optimizer=optimizer, scheduler=compression_scheduler,
                                 extras=checkpoint_extras, is_best=is_best, name=args.name, dir=msglogger.logdir)

    # Finally run results on the test set
    test(test_loader, model, criterion, [pylogger], activations_collectors, args=args)


def update_training_scores_history(perf_scores_history, model, total_loss, classification_loss, regression_loss, epoch,
                                   num_best_scores):
    """ Update the list of top training scores achieved so far, and log the best scores so far"""

    model_sparsity, _, params_nnz_cnt = distiller.model_params_stats(model)
    perf_scores_history.append(distiller.MutableNamedTuple({'params_nnz_cnt': -params_nnz_cnt,
                                                            'sparsity': model_sparsity,
                                                            'total_loss': total_loss,
                                                            'classification_loss': classification_loss,
                                                            'regression_loss': regression_loss,
                                                            'epoch': epoch}))

    # Keep perf_scores_history sorted from best to worst
    # Sort by sparsity as main sort key, then sort by top1, top5 and epoch

    def key(entry):
        return (
            entry['params_nnz_cnt'],
            -entry['total_loss'],
            -entry['classification_loss'],
            -entry['regression_loss'],
            entry['epoch'],
        )

    perf_scores_history.sort(key=key, reverse=True)
    for score in perf_scores_history[:num_best_scores]:
        msglogger.info(
            '==> Best [Total Loss: %.3f   Classification Loss: %.3f   Regression Loss: %.3f   Sparsity:%.2f   Params: %d on epoch: %d]',
            score.total_loss, score.classification_loss, score.regression_loss, score.sparsity, -score.params_nnz_cnt,
            score.epoch)


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

    total_loss, classification_loss, regression_loss = test(
        test_loader, model, criterion, loggers, activations_collectors, args=args)

    if args.quantize_eval:
        checkpoint_name = 'quantized'
        apputils.save_checkpoint(0, args.arch, model, optimizer=None, scheduler=scheduler,
                                 name='_'.join([args.name, checkpoint_name]) if args.name else checkpoint_name,
                                 dir=msglogger.logdir, extras={'quantized_total_loss': total_loss})


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
    msglogger.info(
        'Collecting activation histograms based on {:.1%} of test dataset'.format(args.activation_histograms))
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
