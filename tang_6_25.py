#-*-coding:utf-8-*-
import os
import os.path as osp
import cv2
import csv

# 1 danyu tang-danyu-1-30.csv confidence = 1 , to 枫叶
# 2 chenxinxin and youjun, tang-1-30.csv and tang-31-60.csv, confidence vs bbx size
# 3 tang nms

lines = []
img_names = []
# with open('./1-30.csv', mode='r') as csvfile:
#     reader = csv.reader(csvfile, delimiter=' ')
#     for i in reader:
#         lines.append(i)

# with open('./tang-danyu-1-30.csv', mode='wb') as csvfile:
#     writer = csv.writer(csvfile, delimiter=' ')
#     text = []
#     for line in lines:
#         line[2] = '1.0'
#         text.append(line)
#     for i in text:
#         writer.writerow(i)

lines = open('./test-labels-one-txt/orig_test_label.txt').readlines()
img_name2_bbx = {}
for line in lines:
    line = line.strip()
    name = line.split(' ')[0]
    bbx = [int(i) for i in line.split(' ')[1:]]
    img_name2_bbx[name] = bbx

with open('./tang-1-55.csv', mode='wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for ii in range(1,56,1):
        bbxs_size = []
        bbxs = []
        cur_names = []
        for line in lines:
            line = line.strip()
            if int(line.split(' ')[1])==ii:
                name = line.split(' ')[0]
                cur_names.append(name)
                bbx = [int(i) for i in line.split(' ')[1:]]
                bbxs.append(bbx)
                bbxs_size.append((int(bbx[3])-int(bbx[1]))*(bbx[4])-int(bbx[2]))
        max_size = max(bbxs_size)
        min_size = min(bbxs_size)
        print(max_size-min_size)
        for name,size,bbx in zip(cur_names, bbxs_size, bbxs):
            temp = [name, str(img_name2_bbx[name][0]),\
            str(round(0.2*float(size-min_size)/float(max_size-min_size)+0.8,4)),\
            str(bbx[1]), str(bbx[2]),\
            str(bbx[3]), str(bbx[4])]
            writer.writerow(temp)


with open('./tang-56-60.csv', mode='wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for ii in range(56,61,1):
        bbxs_size = []
        bbxs = []
        cur_names = []
        for line in lines:
            line = line.strip()
            if int(line.split(' ')[1])==ii:
                name = line.split(' ')[0]
                cur_names.append(name)
                bbx = [int(i) for i in line.split(' ')[1:]]
                bbxs.append(bbx)
                bbxs_size.append((int(bbx[3])-int(bbx[1]))*(bbx[4])-int(bbx[2]))
        max_size = max(bbxs_size)
        min_size = min(bbxs_size)
        print(max_size-min_size)
        for name,size,bbx in zip(cur_names, bbxs_size, bbxs):
            temp = [name, str(img_name2_bbx[name][0]),\
            str(round(0.2*float(size-min_size)/float(max_size-min_size)+0.8,4)),\
            str(bbx[1]), str(bbx[2]),\
            str(bbx[3]), str(bbx[4])]
            writer.writerow(temp) 


# import random
# # 模拟计算Map的过程，讨论confidence对于map是否有影响
# # 结论，是有影响的。实际上，与gt越接近的预测框的置信度越高，那么实际测出来的map就越高 => 影响precision
# # 而且提交的预测框的个数越接近于gt框个数，则map越高 => 影响recall
# def voc_ap(rec, prec, use_07_metric=True):
#     """ ap = voc_ap(rec, prec, [use_07_metric])
#     Compute VOC AP given precision and recall.
#     If use_07_metric is true, uses the
#     VOC 07 11 point method (default:True).
#     """
#     if use_07_metric:
#         # 11 point metric
#         ap = 0.
#         for t in np.arange(0., 1.1, 0.1):
#             if np.sum(rec >= t) == 0:
#                 p = 0
#             else:
#                 p = np.max(prec[rec >= t])
#             ap = ap + p / 11.
#     else:
#         # correct AP calculation
#         # first append sentinel values at the end
#         mrec = np.concatenate(([0.], rec, [1.]))
#         mpre = np.concatenate(([0.], prec, [0.]))

#         # compute the precision envelope
#         for i in range(mpre.size - 1, 0, -1):
#             mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

#         # to calculate area under PR curve, look for points
#         # where X axis (recall) changes value
#         i = np.where(mrec[1:] != mrec[:-1])[0]

#         # and sum (\Delta recall) * prec
#         ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#     return ap

# nd = 6639
# npos = 4351
# tp_all = 4351
# fp_all = 2288

# tp_index = range(0,nd,1)

# ap_list = []
# for i in range(1000):
#     tp = np.zeros(nd)
#     fp = np.zeros(nd)
#     cur_tp_index = random.sample(tp_index, tp_all)
#     for index in range(0,nd,1):
#         if index in cur_tp_index:
#             tp[index] = 1.
#         else:
#             fp[index] = 1.
#     fp = np.cumsum(fp)  #  fp累加
#     tp = np.cumsum(tp)
#     rec = tp / float(npos)
#     prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#     ap = voc_ap(rec, prec, use_07_metric=True)
#     ap_list.append(ap)
#     # print(ap)

# print("max ap is: ", round(max(ap_list), 2))  #  ('max ap is: ', 0.7)
# print("min ap is: ", round(min(ap_list), 2))  #  ('min ap is: ', 0.66)

        