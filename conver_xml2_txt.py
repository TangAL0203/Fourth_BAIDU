#-*-coding:utf-8-*-
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
import cv2

class_id2_name = {1:'senda',
2:'haagen_dazs',
3:'dhc',
4:'zara',
5:'xiabuxiabu',
6:'anta',
7:'vivo',
8:'anta_kids',
9:'columbia',
10:'bosideng',
11:'tmj',
12:'mido',
13:'tebu',
14:'zmn',
15:'samsung',
16:'jwyb',
17:'lyf',
18:'baleno',
19:'happy_lemon',
20:'burger_king',
21:'oppo',
22:'ajidou',
23:'zhouheiya',
24:'xyx',
25:'hla',
26:'vans',
27:'st_sat',
28:'uniqlo',
29:'puma',
30:'bsk',
31:'camel',
32:'coco',
33:'gujin',
34:'wqlm',
35:'hm,hm',
36:'gong_cha',
37:'pierre_cardin',
38:'calvin_klein',
39:'huawei',
40:'innisfree',
41:'maybelline',
42:'converse',
43:'la_chapelle',
44:'new_balance',
45:'li_ning',
46:'peacebird',
47:'playboy',
48:'youngor',
49:'chando',
50:'herborist',
51:'jack_jones',
52:'selected',
53:'xbk',
54:'mdl',
55:'belle',
56:'vero_moda',
57:'watsons',
58:'kfc',
59:'nike',
60:'adidas'}

colors = [(255,182,193),
(255,192,203),
(220,20,60),
(255,240,245),
(219,112,147),
(255,105,180),
(255,20,147),
(199,21,133),
(218,112,214),
(216,191,216),
(221,160,221),
(238,130,238),
(255,0,255),
(255,0,255),
(139,0,139),
(128,0,128),
(186,85,211),
(148,0,211),
(153,50,204),
(75,0,130),
(138,43,226),
(147,112,219),
(123,104,238),
(106,90,205),
(72,61,139),
(230,230,250),
(248,248,255),
(0,0,255),
(0,0,205),
(25,25,112),
(0,0,139),
(0,0,128),
(65,105,225),
(100,149,237),
(176,196,222),
(119,136,153),
(112,128,144),
(30,144,255),
(240,248,255),
(70,130,180),
(135,206,250),
(135,206,235),
(0,191,255),
(173,216,230),
(176,224,230),
(95,158,160),
(240,255,255),
(225,255,255),
(175,238,238),
(0,255,255),
(0,255,255),
(0,206,209),
(47,79,79),
(0,139,139),
(0,128,128),
(72,209,204),
(32,178,170),
(64,224,208),
(127,255,170),
(0,250,154)]

# get xml file and convert them into txt label and single txt file

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0 # bbx的x中心点坐标
    y = (box[2] + box[3])/2.0 # bbx的y中心点坐标
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw # 归一化中心点坐标x
    w = w*dw # 归一化bbx宽
    y = y*dh # 归一化中心点坐标y
    h = h*dh # 归一化bbx高
    return (x,y,w,h)

def convert_annotation(image_id, f):
    in_file = open('./test-xml/%s.xml'%(image_id))
    out_file = open('./test-labels/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls_id = int(obj.find('name').text.split(',')[0])
        if cls_id not in range(1,61,1):
            continue
        xmlbox = obj.find('bndbox')
        bb = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        f.write(image_id+'.jpg '+str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

image_ids = os.listdir('./test')
f = open('./test-labels-one-txt/orig_test_label.txt', 'w')
for image_id in image_ids:
    print(image_id.split('.')[0])
    convert_annotation(image_id.split('.')[0], f)
f.close()


lines = open('./test-labels-one-txt/orig_test_label.txt').readlines()

img_id2_box = {}
for name in os.listdir('./test/'):
    img_id2_box[name] = []

for line in lines:
    line = line.strip()
    img_name = line.split(' ')[0]
    img_id2_box[img_name].append([max(int(i.split('.')[0]),0) for i in line.split(' ')[1:]])

for name in os.listdir('./test/'):
    img = cv2.imread('./test/'+name)
    for bbx in img_id2_box[name]:
        print(bbx)
        bbx[1] = max(int(bbx[1]), 0)
        bbx[2] = max(int(bbx[2]), 0)
        bbx[3] = max(int(bbx[3]), 0)
        bbx[4] = max(int(bbx[4]), 0)
        pt1 = (bbx[1],bbx[2])
        pt2 = (bbx[3],bbx[4])
        color = colors[int(bbx[0])-1][::-1]
        cv2.putText(img,
                class_id2_name[int(bbx[0])],
                pt1,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),2,cv2.LINE_AA)
        cv2.rectangle(img, pt1, pt2, color, thickness=3)

    cv2.imwrite('./orig-test-box/'+name, img)



import random
# 模拟计算Map的过程，讨论confidence对于map是否有影响
# 结论，是有影响的。实际上，与gt越接近的预测框的置信度越高，那么实际测出来的map就越高 => 影响precision
# 而且提交的预测框的个数越接近于gt框个数，则map越高 => 影响recall
def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

nd = 500
npos = 500
tp_all = 450
fp_all = 50

tp_index = range(0,500,1)

ap_list = []
for i in range(1000):
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    cur_tp_index = random.sample(tp_index, tp_all)
    for index in range(1,500,1):
        if index in cur_tp_index:
            tp[index] = 1.
        else:
            fp[index] = 1.
    fp = np.cumsum(fp)  #  fp累加
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric=True)
    ap_list.append(ap)
    # print(ap)

print("max ap is: ", round(max(ap_list), 2))
print("min ap is: ", round(min(ap_list), 2))

