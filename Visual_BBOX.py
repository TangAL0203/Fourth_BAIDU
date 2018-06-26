#-*-coding:utf-8-*-
import os
import cv2
import numpy as np

# img_names = os.listdir('./train')
# BBXS = {}
# for name in img_names:
#     BBXS[name] = []
# RGB_Values = [(255,182,193),
#             (255,192,203),
#             (220,20,60),
#             (255,240,245),
#             (219,112,147),
#             (255,105,180),
#             (255,20,147),
#             (199,21,133),
#             (218,112,214),
#             (216,191,216),
#             (221,160,221),
#             (238,130,238),
#             (255,0,255),
#             (255,0,255),
#             (139,0,139),
#             (128,0,128),
#             (186,85,211),
#             (148,0,211),
#             (153,50,204),
#             (75,0,130),
#             (138,43,226),
#             (147,112,219),
#             (123,104,238),
#             (106,90,205),
#             (72,61,139),
#             (230,230,250),
#             (248,248,255),
#             (0,0,255),
#             (0,0,205),
#             (25,25,112),
#             (0,0,139),
#             (0,0,128),
#             (65,105,225),
#             (100,149,237),
#             (176,196,222),
#             (119,136,153),
#             (112,128,144),
#             (30,144,255),
#             (240,248,255),
#             (70,130,180),
#             (135,206,250),
#             (135,206,235),
#             (0,191,255),
#             (173,216,230),
#             (176,224,230),
#             (95,158,160),
#             (240,255,255),
#             (225,255,255),
#             (175,238,238),
#             (0,255,255),
#             (0,255,255),
#             (0,206,209),
#             (47,79,79),
#             (0,139,139),
#             (0,128,128),
#             (72,209,204),
#             (32,178,170),
#             (64,224,208),
#             (127,255,170),
#             (0,250,154)]

# lines = open('./train.txt').readlines()
# # label from 1 to 60
# for line in lines:
#     line = line.strip()
#     img_name = line.split(' ')[0]
#     bbx = [int(i) for i in line.split(' ')[1:]]
#     BBXS[img_name].append(bbx)

# for name in img_names:
#     print(name)
#     # img = cv2.imdecode(np.fromfile('./train/'+name,dtype=np.uint8),-1)
#     img = cv2.imread('./train/'+name)
#     bbxs = BBXS[name]
#     for bbx in bbxs:
#         color = RGB_Values[bbx[0]-1][::-1]
#         pt1 = (bbx[1], bbx[2])
#         pt2 = (bbx[3], bbx[4])
#         cv2.rectangle(img, pt1, pt2, color, thickness=3)
#     print(img.shape)
#     # print(cv2.imencode('.jpg',img)[1].tofile('./Visual/train/'+name))
#     print(cv2.imwrite(r'E:\\Fourth_Bear\\repecharge\\datasets\\Visual\\train\\'+name, img))


## plt train label hist figure
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)

# lines = open('./train.txt').readlines()
# # label from 1 to 60
# labels = []
# for line in lines:
#     line = line.strip()
#     img_name = line.split(' ')[0]
#     bbx = [int(i) for i in line.split(' ')[1:]]
#     labels.append(bbx[0])

# # hist只能统计一个范围的数据，划分为bins个区间，各个区间的统计量的直方图
# n, bins, patches = ax.hist(labels, bins=60) # bins 区间个数

# print(bins)
# plt.title('Num vs Label id', fontsize=20)
# plt.xlabel('Labe id', fontsize=20)
# plt.ylabel('Num', fontsize=20)
# plt.show()

# import os
# import shutil

# lines = open('./test.txt').readlines()
# lines = lines[1750:2600]
# for line in lines:
#     line = line.strip()
#     print(lin)
#     src = './test/'+line
#     dist = './tang_test/'+line
#     shutil.copy(src, dist)

# label_names = ['1_SENDA森达',
# '2_Haagen_Dazs',
# '3_DHC',
# '4_ZARA',
# '5_呷哺呷哺xiabuxiabu',
# '6_安踏体育ANTA SPORTS',
# '7_vivo',
# '8_kids安踏儿童',
# '9 _Columbia',
# '10_BOSIDENG,波司登',
# '11_谭木匠',
# '12_MIDO美度',
# '13_X特步',
# '14_TUCANO啄木鸟',
# '15_SAMSUNG',
# '16_绝味鸭脖',
# '17_来伊份lyfen',
# '18_Baleno班尼路',
# '19_happy_lemon快乐柠檬',
# '20_BURGER KING汉堡王',
# '21_oppo',
# '22_AJIDOU阿吉豆',
# '23_ZHOUHEIYA,周黑鸭',
# '24_鲜芋仙',
# '25_HLA海澜之家',
# '26_VANS',
# '27_ST&SAT',
# '28_UNIQLO',
# '29_PUMA',
# '30_必胜客PizzaHut',
# '31_CAMEL',
# '32_CoCo都可',
# '33_GUJIN古今',
# '34_味千拉面',
# '35_H.M',
# '36_GONGCHA贡茶',
# '37_pierre_cardin皮尔卡丹',
# '38_Calvin Klein',
# '39_HUAWEI',
# '40_innisfree',
# '41_MAYBELLINE',
# '42_CONVERSE,匡威',
# '43_La Chapelle',
# '44_new_balance',
# '45_李宁',
# '46_PEACEBIRD',
# '47_PLAYBOY',
# '48_YOUNGOR 雅戈尔',
# '49_CHANDO自然堂',
# '50_HERBORIST,佰草集',
# '51_JACKJONES',
# '52_SELECTED',
# '53_星巴克',
# '54_麦当劳',
# '55_BeLLE',
# '56_VERO MODA',
# '57_watsons,屈臣氏',
# '58_kfc,肯德基',
# '59_NIKE',
# '60_adidas']

# import shutil
# import os
# for name in label_names:
#     if not os.path.exists('./Visual/train/'+name):
#         os.mkdir('./Visual/train/'+name)

# lines = open('./train.txt').readlines()
# # label from 1 to 60
# for line in lines:
#     line = line.strip()
#     img_name = line.split(' ')[0]
#     print(img_name)
#     bbx = [int(i) for i in line.split(' ')[1:]]
#     src = './Visual/train/'+img_name
#     dist = './Visual/train/'+label_names[bbx[0]-1]+'/'+img_name
#     shutil.copy(src, dist)

import os
import shutil
lines = os.listdir('./tang_test')
jpg_names = []
xml_names = []
for line in lines:
    line = line.strip()
    if line.endswith('jpg'):
        jpg_names.append(line)
    elif line.endswith('xml'):
        xml_names.append(line.split('.')[0]+'.jpg')

zyj = []

'./tang_test/zrj/'
f = open('./zrj.txt', 'w')
count = 0
for jpg_name in jpg_names:
    if jpg_name not in xml_names and count<=399:
        print(jpg_name)
        zyj.append(jpg_name)
        f.write(jpg_name+'\n')
        src = './tang_test/'+jpg_name
        dist = './tang_test/zrj/'+jpg_name
        shutil.copy(src, dist)
        os.remove(src)
        count+=1

f.close()

