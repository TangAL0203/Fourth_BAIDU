#-*-coding:utf-8-*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import functional as F
import utils.dataset as dataset
import utils.models as models
from utils.train_Components import *
import torchvision
import torchvision.transforms as transforms
import shutil
import math
import os
import os.path as osp
import math
import argparse
from collections import OrderedDict

def get_args():
    parser = argparse.ArgumentParser(description='Fourth Baidu Competition Experiment')

    parser.add_argument('--arch', metavar='ARCH', default='Resnet50', help='model architecture')
    parser.add_argument('--gpuId', default='0', type=str, help='GPU Id')
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--data_path', metavar='DATA_PATH', type=str, default=['./train', './val', './test'],
                        help='path to train val and test dataset', nargs=2)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--savePath', default='./models', type=str, \
                        help='path to save model')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--zeroTrain', default=False, action='store_true', help='choose if train from Scratch or not')
    parser.add_argument('--TenCrop', default=False, action='store_true', help='choose if using TenCrop on test imgs')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')


    args = parser.parse_args()
    return args

args = get_args()

print("arch         is: {}".format(args.arch))
print("gpuId        is: {}".format(args.gpuId))
print("init lr      is: {}".format(args.lr))
print("batch size   is: {}".format(args.batch_size))
print("epochs       is: {}".format(args.epochs))
print("savePath     is: {}".format(args.savePath))
print("resume       is: {}".format(args.resume))
print("momentum     is: {}".format(args.momentum))
print("zeroTrain    is: {}".format(args.zeroTrain))
print("TenCrop      is: {}".format(args.TenCrop))
print("weight_decay is: {}".format(args.weight_decay))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuId

use_gpu = torch.cuda.is_available()
num_batches = 0

def train_val_test(model, train_loader, val_loader, test_loader, print_freq=50, TenCroptest_loader=None, optimizer=None, epoches=10):
    global args, num_batches
    print("Start training.")
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    StepLr = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    max_val_acc = 0
    max_test_acc = 0
    for i in range(epoches):
        if i<=25:
            StepLr.step(i)
        model.train()
        print("Epoch: ", i, "lr is: {}".format(StepLr.get_lr()))
        num_batches = train_epoch(model, num_batches, train_loader, print_freq=print_freq, optimizer=optimizer)
        if not args.TenCrop:
            cur_val_acc, cur_test_acc = get_train_val_test_acc(model, train_loader, val_loader, test_loader)
        else:
            cur_val_acc, cur_test_acc = get_train_val_testTenCrop_acc(model, train_loader, val_loader, test_loader, TenCroptest_loader)
        if i==0:
            max_val_acc, max_test_acc = cur_val_acc, cur_test_acc
            filename = "{}_{}_{}.pth".format(args.arch, str(cur_val_acc), str(cur_test_acc))
            torch.save(model.state_dict(), osp.join(args.savePath, filename))
        elif max_test_acc<cur_test_acc:
            # delete old state_dict
            old_filename = "{}_{}_{}.pth".format(args.arch, str(max_val_acc), str(max_test_acc))
            os.remove(osp.join(args.savePath, old_filename))
            max_val_acc, max_test_acc = cur_val_acc, cur_test_acc
            filename = "{}_{}_{}.pth".format(args.arch, str(cur_val_acc), str(cur_test_acc))
            torch.save(model.state_dict(), osp.join(args.savePath, filename))

    print("Finished training.")

def main():
    global args, num_batches, use_gpu
    if not args.zeroTrain:
        if args.arch == "densenet201":
            model = models.Modified_densenet201()
        elif args.arch == "Resnet50":
            model = models.Modified_Resnet50()
        elif args.arch == "Resnet101":
            model = models.Modified_Resnet101()
        elif args.arch == "Resnet152":
            model = models.Modified_Resnet152()
        elif args.arch == "nasnetalarge":
            model = models.Modified_nasnetalarge()

        if args.resume:
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint, strict=True)

        if use_gpu:
            model = model.cuda()
            print("Use GPU!")
        else:
            print("Use CPU!")

        train_path = args.data_path[0]
        val_path = args.data_path[1]
        test_path = args.data_path[2]

        train_loader = dataset.train_loader(train_path, batch_size=args.batch_size, num_workers=10, pin_memory=True)
        val_loader = dataset.test_loader(val_path, batch_size=1, num_workers=10, pin_memory=True)
        test_loader = dataset.test_loader(test_path, batch_size=1, num_workers=10, pin_memory=True)

        if args.TenCrop:
            TenCroptest_loader = dataset.testTenCrop_loader(test_path, batch_size=1, num_workers=10, pin_memory=True)
        else:
            TenCroptest_loader =None

    train_val_test(model, train_loader, val_loader, test_loader, print_freq=50, TenCroptest_loader=TenCroptest_loader, optimizer=None, epoches=args.epochs)

if __name__ == "__main__":
    main()