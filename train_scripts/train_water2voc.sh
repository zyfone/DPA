#!/bin/bash
save_dir="./weight_model/water2voc/"
dataset="water2voc"
pretrained_path="/root/pretrained_model/resnet101_caffe.pth"
net="res101_fix"

CUDA_VISIBLE_DEVICES=4 python -u da_train_net.py \
--max_epochs 10 --cuda --dataset ${dataset} \
--net ${net} --save_dir ${save_dir} \
--pretrained_path ${pretrained_path} \
--gc --lc --da_use_contex \
--weight_consis 0.1 --lr_bound 0.5 --gmm_split 0.01 --instance_da_eta 0.01

CUDA_VISIBLE_DEVICES=4 python test_scripts/test_water2voc.py
