#-*-coding:utf-8-*-
import os
import os.path as osp
import shutil

for i in range(1,101):
    if not os.path.exists(osp.join('./train', str(i))):
        os.makedirs(osp.join('./train', str(i)))

for i in range(1,101):
    if not os.path.exists(osp.join('./val', str(i))):
        os.makedirs(osp.join('./val', str(i)))

train_ratio = 0.8

for i in range(1,101):
    names = os.listdir(osp.join('./orig_train',str(i)))
    train_names = names[0:int(len(names)*train_ratio)]
    val_names = names[int(len(names)*train_ratio):]
    for name in train_names:
        shutil.copy(osp.join('./orig_train',str(i),name), osp.join('./train', str(i)))
    for name in val_names:
        shutil.copy(osp.join('./orig_train',str(i),name), osp.join('./val', str(i)))


