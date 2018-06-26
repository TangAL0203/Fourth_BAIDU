#-*-coding:utf-8-*-
import cv2
import os
import os.path as osp
import csv
import shutil

all_img_names = os.listdir('./test')

img_names = []
with open('./tang_second_test.csv', mode='r') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for i in reader:
        if i[0] not in img_names:
            img_names.append(i[0])

for name in all_img_names:
    if name not in img_names:
        shutil.copy('./test/'+name, './fuck_result/miss_img/'+name)