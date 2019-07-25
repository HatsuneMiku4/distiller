#!/usr/bin/env bash
# pretrained path: ~/.cache/torch/checkpoints/

# FUNCTION: Evaluate pretrained mobilenetv2_ssdlite on voc2012 test set
# TODO: add --pretrained support
VOC2012_DIR=~/data/VOC2012/
python compress_detector.py ${VOC2012_DIR} --name eval_mobilenetv2_ssdlite_voc2012 \
--arch mobilenet_v2 --evaluate \
--batch-size 32  --gpus 1,2,3 --workers 32 \
--print-freq 10

VOC2012_DIR=~/data/VOC2012/
python compress_detector.py ${VOC2012_DIR} --name eval_mobilenetv2_ssdlite_voc2012 \
--arch mobilenet_v2 --pretrained --evaluate \
--batch-size 32  --gpus 1,2,3 --workers 32 \
--print-freq 10


# FUNCTION: Export pretrained mobilenetv2 to onnx
# ERROR: ValueError: only one element tensors can be converted to Python scalars
# SOLUTION: https://github.com/pytorch/pytorch/issues/20516#issuecomment-498102525
VOC2012_DIR=~/data/VOC2012/
python compress_detector.py ${VOC2012_DIR} --name export_mobilenetv2_imagenet \
--arch mobilenet_v2 --gpus 1,2,3 \
--export-onnx export.onnx

# FUNCTION: Print model summary of pretrained mobilenetv2
VOC2012_DIR=~/data/VOC2012/
python compress_detector.py ${VOC2012_DIR} --name summary_mobilenetv2_ssdlite_voc2012 \
--arch mobilenet_v2 --gpus 3 \
--summary compute --summary model --summary modules

# FUNCTION: Post-training quantize (linearly) and evaluate mobilenetv2 on imagenet
# --qe-mode: sym, asym_s, asym_u
# --qe-per-channel, --qe-config-file, --qe-stats-file
# TODO: --qe-scale-approx-bits
VOC2012_DIR=~/data/VOC2012/
python compress_detector.py ${VOC2012_DIR} --name post_quant_eval_mobilenetv2_ssdlite_voc2012 \
--arch mobilenet_v2 --gpus 1,2,3 \
--quantize-eval --evaluate --qe-mode asym_u \
--qe-bits-acts 8 --qe-bits-wts 8 --qe-bits-accum 32 \
--qe-clip-acts avg

VOC2012_DIR=~/data/VOC2012/
python compress_detector.py ${VOC2012_DIR} --name post_quant_eval_mobilenetv2_ssdlite_voc2012 \
--arch mobilenet_v2 --pretrained --calculate-map --gpus 3 \
--quantize-eval --evaluate --qe-config-file ./mb2_ssdlite_voc2012_post_train.yaml

# FUNCTION: Quant-aware train mobilenetv2 ssdlite on voc2012
VOC2012_DIR=~/data/VOC2012/
python compress_detector.py ${VOC2012_DIR} --name quant_aware_mobilenetv2_ssdlite_voc2012 \
--arch mobilenet_v2 --pretrained --gpus 0 --workers 32 \
--batch-size 64 --epochs 20 --learning-rate 0.001 \
--compress ../quantization/quant_aware_train/quant_aware_train_linear_quant.yaml \
--print-freq 10


# FUNCTION: sensitivity analysis
VOC2012_DIR=~/data/VOC2012/
python compress_detector.py ${VOC2012_DIR} --name sensitivity_analysis_mobilenetv2_ssdlite_voc2012 \
--arch mobilenet_v2 --evaluate --sense filter \
--batch-size 32  --gpus 1,2,3 --workers 32 \
--print-freq 10