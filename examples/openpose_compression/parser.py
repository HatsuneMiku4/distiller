#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse

import distiller
import distiller.quantization
import examples.automated_deep_compression as adc
from distiller.utils import float_range_argparse_checker as float_range
import distiller.models as models


SUMMARY_CHOICES = ['sparsity', 'compute', 'model', 'modules', 'png', 'png_w_params']
ALL_MODEL_NAMES = ['shufflenetv2', 'hourglass', 'vgg19']


def get_parser():
    parser = argparse.ArgumentParser(description='Distiller pose estimation model compression')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--anno-dir', dest='anno_dir', metavar='DIR', help='path to annotation files', default='/home/CORP.PKUSC.ORG/hatsu3/data/coco/annotations2014')
    parser.add_argument('--json-path', dest='json_path', metavar='PATH', help='path to json file', default='/home/CORP.PKUSC.ORG/hatsu3/data/coco/etc/COCO.json')
    parser.add_argument('--mask-dir', dest='mask_dir', metavar='DIR', help='path to mask dir', default='/home/CORP.PKUSC.ORG/hatsu3/data/coco/etc/mask')
    parser.add_argument('--image-list', dest='image_list_txt', metavar='PATH', help='path to image list file', default='/home/CORP.PKUSC.ORG/hatsu3/research/lab_projects/openpose/pytorch_openpose/evaluate/image_info_val2014_1k.txt')
    parser.add_argument('--vis-dir', dest='vis_dir', metavar='DIR', help='path to visualization dir', default='/home/CORP.PKUSC.ORG/hatsu3/research/lab_projects/openpose/pytorch_openpose/evaluate/vis')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19', type=lambda s: s.lower(),
                        choices=ALL_MODEL_NAMES, help='model architecture: ' + ' | '.join(ALL_MODEL_NAMES) + ' (default: vgg19)')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', type=int, metavar='N', help='number of total epochs to run (default: 90')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')

    optimizer_args = parser.add_argument_group('Optimizer arguments')
    optimizer_args.add_argument('--lr', '--learning-rate', default=1, type=float, metavar='LR', help='initial learning rate')
    optimizer_args.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    optimizer_args.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

    load_checkpoint_group = parser.add_argument_group('Resuming arguments')
    load_checkpoint_group_exc = load_checkpoint_group.add_mutually_exclusive_group()
    load_checkpoint_group_exc.add_argument('--resume-from', dest='resumed_checkpoint_path', default='', type=str, metavar='PATH', help='path to latest checkpoint. Use to resume paused training session.')
    load_checkpoint_group_exc.add_argument('--exp-load-weights-from', dest='load_model_path', default='', type=str, metavar='PATH', help='path to checkpoint to load weights from (excluding other fields) (experimental)')
    load_checkpoint_group.add_argument('--load-vgg19', dest='load_vgg19', metavar='PATH', help='path to pretrained vgg19-openpose')
    load_checkpoint_group.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    load_checkpoint_group.add_argument('--reset-optimizer', action='store_true', help='Flag to override optimizer if resumed from checkpoint. This will reset epochs count.')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')
    parser.add_argument('--activation-stats', '--act-stats', nargs='+', metavar='PHASE', default=list(),
                        help='collect activation statistics on phases: train, valid, and/or test'
                        ' (WARNING: this slows down training)')
    parser.add_argument('--activation-histograms', '--act-hist',
                        type=distiller.utils.float_range_argparse_checker(exc_min=True),
                        metavar='PORTION_OF_TEST_SET',
                        help='Run the model in evaluation mode on the specified portion of the test dataset and '
                             'generate activation histograms. NOTE: This slows down evaluation significantly')
    parser.add_argument('--masks-sparsity', dest='masks_sparsity', action='store_true', default=False,
                        help='print masks sparsity table at end of each epoch')
    parser.add_argument('--param-hist', dest='log_params_histograms', action='store_true', default=False,
                        help='log the parameter tensors histograms to file (WARNING: this can use significant disk space)')
    parser.add_argument('--summary', type=lambda s: s.lower(), choices=SUMMARY_CHOICES, action='append',
                        help='print a summary of the model, and exit - options: | '.join(SUMMARY_CHOICES))
    parser.add_argument('--export-onnx', action='store', nargs='?', type=str, const='model.onnx', default=None,
                        help='export model to ONNX format')
    parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                        help='configuration file for pruning the model (default is to use hard-coded schedule)')
    parser.add_argument('--sense', dest='sensitivity', choices=['element', 'filter', 'channel'], type=lambda s: s.lower(),
                        help='test the sensitivity of layers to pruning')
    parser.add_argument('--sense-range', dest='sensitivity_range', type=float, nargs=3, default=[0.0, 0.95, 0.05],
                        help='an optional parameter for sensitivity testing providing the range of sparsities to test.\n'
                        'This is equivalent to creating sensitivities = np.arange(start, stop, step)')
    parser.add_argument('--extras', default=None, type=str,
                        help='file with extra configuration information')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed the PRNG for CPU, CUDA, numpy, and Python')
    parser.add_argument('--deterministic', '--det', action='store_true',
                        help='Ensure deterministic execution for re-producible results.')
    parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU only. \n'
                        'Flag not set => uses GPUs according to the --gpus flag value.'
                        'Flag set => overrides the --gpus flag')
    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump logs and checkpoints')
    parser.add_argument('--num-best-scores', dest='num_best_scores', default=1, type=int,
                        help='number of best scores to track and report (default: 1)')
    parser.add_argument('--load-serialized', dest='load_serialized', action='store_true', default=False,
                        help='Load a model without DataParallel wrapping it')
    parser.add_argument('--thinnify', dest='thinnify', action='store_true', default=False,
                        help='physically remove zero-filters and create a smaller model')

    distiller.quantization.add_post_train_quant_args(parser)
    distiller.pruning.greedy_filter_pruning.add_greedy_pruner_args(parser)
    adc.automl_args.add_automl_args(parser)
    return parser
