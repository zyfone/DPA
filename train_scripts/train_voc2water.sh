#!/bin/bash
save_dir="./weight_model/voc2water/"
dataset="voc2water"
pretrained_path="/root/pretrained_model/resnet101_caffe.pth"
net="res101"

CUDA_VISIBLE_DEVICES=3 python -u da_train_net.py \
--max_epochs 10 --cuda --dataset ${dataset} \
--net ${net} --save_dir ${save_dir} \
--pretrained_path ${pretrained_path} \
--gc --lc --da_use_contex \
--weight_consis 0.1 --lr_bound 0.1 --gmm_split 0.01 \
--dropout_consis

CUDA_VISIBLE_DEVICES=3 python  test_scripts/test_voc2water.py