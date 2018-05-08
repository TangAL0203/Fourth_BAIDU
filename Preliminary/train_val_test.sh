#!/usr/bin/env sh
echo "train baidu Preliminary dataset, train:val = 8:2"
echo "arch: densenet201, conv_init_lr: 0.001, fc_init_lr: 0.001"
echo "using pytorch data augmentation"
echo "using tencrop testing"
python train_val_test.py --arch Resnet50 --batch_size 8 --epochs 50 --gpuId 0 --momentum 0.9 --weight_decay 1e-4 --print_freq 50 --TenCrop

