#!/usr/bin/env bash
# pretrained path: ~/.cache/torch/checkpoints/

# FUNCTION: Evaluate pretrained mobilenetv2 on imagenet test set
IMAGENET_DIR=~/data/imagenet/raw-data/
python compress_classifier.py ${IMAGENET_DIR} --name eval_mobilenetv2_imagenet \
--arch mobilenet_v2 --pretrained --evaluate \
--batch-size 256  --gpus 1,2,3 --workers 32 \
--print-freq 10

# FUNCTION: Export pretrained mobilenetv2 to onnx
# ERROR: ValueError: only one element tensors can be converted to Python scalars
# SOLUTION: https://github.com/pytorch/pytorch/issues/20516#issuecomment-498102525
IMAGENET_DIR=~/data/imagenet/raw-data/
python compress_classifier.py ${IMAGENET_DIR} --name export_mobilenetv2_imagenet \
--arch mobilenet_v2 --pretrained --gpus 1,2,3 \
--export-onnx export.onnx

# FUNCTION: Print model summary of pretrained mobilenetv2
IMAGENET_DIR=~/data/imagenet/raw-data/
python compress_classifier.py ${IMAGENET_DIR} --name summary_mobilenetv2_imagenet \
--arch mobilenet_v2 --pretrained --gpus 1,2,3 \
--summary sparsity --summary compute --summary model --summary modules

# FUNCTION: Post-training quantize (linearly) and evaluate mobilenetv2 on imagenet
# --qe-mode: sym, asym_s, asym_u
# --qe-per-channel, --qe-config-file, --qe-stats-file
# TODO: --qe-scale-approx-bits
IMAGENET_DIR=~/data/imagenet/raw-data/
python compress_classifier.py ${IMAGENET_DIR} --name post_quant_eval_mobilenetv2_imagenet \
--arch mobilenet_v2 --pretrained --gpus 1,2,3 \
--quantize-eval --evaluate --qe-mode asym_u \
--qe-bits-acts 8 --qe-bits-wts 8 --qe-bits-accum 32 \
--qe-clip-acts avg

# FUNCTION: Quant-aware train mobilenetv2 on imagenet
IMAGENET_DIR=~/data/imagenet/raw-data/
python compress_classifier.py ${IMAGENET_DIR} --name quant_aware_mobilenetv2_imagenet \
--arch mobilenet_v2 --pretrained \
--gpus 1,2,3 --workers 32 \
--batch-size 256 --epochs 20 --learning-rate 0.1 \
--compress ../quantization/quant_aware_train/quant_aware_train_linear_quant.yaml \
--print-freq 50

# FUNCTION: AMC
IMAGENET_DIR=~/data/imagenet/raw-data/
python compress_classifier.py ${IMAGENET_DIR} --name quant_aware_mobilenetv2_imagenet \
--arch mobilenet_v2 --pretrained \
--gpus 1,2,3 --workers 32 \
--batch-size 256 --epochs 20 --learning-rate 0.1 \
--compress ../quantization/quant_aware_train/quant_aware_train_linear_quant.yaml \
--print-freq 50  \
--amc-agent-algo DDPG --amc-protocol accuracy-guaranteed \
--amc-ft-epochs 1 --amc-heatup-epochs 100 --amc-training-epochs 300


