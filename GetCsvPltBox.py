#-*-coding:utf-8-*-
import os
import os.path as osp
import cv2
import csv


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


class_file_names['1,senda',
'2,haagen_dazs',
'3,dhc',
'4,zara',
'5,xiabuxiabu',
'6,anta',
'7,vivo',
'8,anta_kids',
'9,columbia,',
'10,bosideng',
'11,tmj',
'12,mido',
'13,tebu',
'14,zmn',
'15,samsung',
'16,jwyb',
'17,lyf',
'18,baleno',
'19,happy_lemon',
'20,burger_king',
'21,oppo',
'22,ajidou',
'23,zhouheiya',
'24,xyx',
'25,hla',
'26,vans',
'27,st_sat',
'28,uniqlo',
'29,puma',
'30,bsk',
'31,camel',
'32,coco',
'33,gujin',
'34,wqlm',
'35,hm,hm',
'36,gong_cha',
'37,pierre_cardin',
'38,calvin_klein',
'39,huawei',
'40,innisfree',
'41,maybelline',
'42,converse',
'43,la_chapelle',
'44,new_balance',
'45,li_ning',
'46,peacebird',
'47,playboy',
'48,youngor',
'49,chando',
'50,herborist',
'51,jack_jones',
'52,selected',
'53,xbk',
'54,mdl',
'55,belle',
'56,vero_moda',
'57,watsons',
'58,kfc',
'59,nike',
'60,adidas']

name2_class_id = {}
for id,name in class_id2_name.items():
    name2_class_id[id] = name

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

lines = []
img_names = []
with open('./tang_second_test.csv', mode='r') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for i in reader:
        lines.append(i)
        if i[0] not in img_names:
            img_names.append(i[0])
            
img_id2_box = {}
for name in img_names:
    img_id2_box[name] = []

for line in lines:
    img_name = line[0]
    img_id2_box[img_name].append(line[1:])

for class_name in class_file_names:
    if not os.path.exists('./Visual/orig-test-box/'+class_name):
        os.mkdir('./Visual/orig-test-box/'+class_name)




for name in img_names:
    img = cv2.imread('./test/'+name)
    for bbx in img_id2_box[name]:
        print(bbx)
        bbx[2] = max(int(bbx[2]), 0)
        bbx[3] = max(int(bbx[3]), 0)
        bbx[4] = max(int(bbx[4]), 0)
        bbx[5] = max(int(bbx[5]), 0)
        pt1 = (bbx[2],bbx[3])
        pt2 = (bbx[4],bbx[5])
        color = colors[int(bbx[0])-1][::-1]
        cv2.putText(img,
                class_id2_name[int(bbx[0])]+' '+str(round(float(bbx[1]),2)),
                pt1,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),2,cv2.LINE_AA)
        cv2.rectangle(img, pt1, pt2, color, thickness=3)

    cv2.imwrite('./fuck_result/nms_p_0.5/'+name, img)