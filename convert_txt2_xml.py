# -*- coding: utf-8 -*-
'''
xml格式
    annotation
        folder
        filename
        path
        source
        size
            width
            height
            depth
        segmented
        object1
            name eg clothes.
            pose eg Unspecified.
            truncated eg 0.
            difficult eg 0.
            bndbox
                xmin eg 176.
                ymin eg 129.
                xmax eg 347.
                ymax eg 383.
        object2
            name
            pose
            truncated
            difficult
            bndbox
                xmin
                ymin
                xmax
                ymax
'''

import os
import os.path as osp
import sys
import cv2
from itertools import islice
from xml.dom.minidom import Document
import numpy as np

BbxIndexName = {0: 'senda', 1: 'hgds', 2: 'dhc', 3: 'zara', 4: 'xbxb', 5: 'anta', \
                6: 'vivo', 7: 'anta_kids', 8: 'columbia', 9: 'bosideng', 10: 'tmj', \
                11: 'mido', 12: 'tebu', 13: 'zmn', 14: 'samsung', 15: 'jwyb', \
                16: 'lyf', 17: 'baleno', 18: 'happy_lemon', 19: 'burger_king', 20: 'oppo', \
                21: 'ajidou', 22: 'zhouheiya', 23: 'xyx', 24: 'hla', 25: 'vans', \
                26: 'st_sat', 27: 'uniqlo', 28: 'puma', 29: 'bsk', 30: 'camel', \
                31: 'coco', 32: 'gujin', 33: 'wqlm', 34: 'hm_hm', 35: 'gong_cha', \
                36: 'pierre_cardin', 37: 'calvin_klein', 38: 'huawei', 39: 'innisfree', 40: 'maybelline', \
                41: 'converse', 42: 'la_chapelle', 43: 'new_balance', 44: 'li_ning', 45: 'peacebird', \
                46: 'playboy', 47: 'youngor', 48: 'chando', 49: 'herborist', 50: 'jack_jones', \
                51: 'selected', 52: 'xbk', 53: 'mdl', 54: 'belle', 55: 'vero_moda', \
                56: 'watsons', 57: 'kfc', 58: 'nike', 59: 'adidas'}

def isValidBox(bbx, width, height):
    xmin = bbx[1]
    ymin = bbx[2]
    xmax = bbx[3]
    ymax = bbx[4]

    if xmin>=width or xmax<=0 or ymin>=height or ymax<=0 or width<=0 or height<=0:
        return False
    else:
        return True

'''
#=====Object example:=======
    <object>
        <name>face</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{{}}</xmin>
            <ymin>{{}}</ymin>
            <xmax>{{}}</xmax>
            <ymax>{{}}</ymax>
        </bndbox>
    </object>
'''
def insertObject(doc, bbx):
    obj = doc.createElement('object')
    name = doc.createElement('name')
    name.appendChild(doc.createTextNode(BbxIndexName[bbx[0]]))
    obj.appendChild(name)
    pose = doc.createElement('pose')
    pose.appendChild(doc.createTextNode('Unspecified'))
    obj.appendChild(pose)
    truncated = doc.createElement('truncated')
    truncated.appendChild(doc.createTextNode(str(0)))
    obj.appendChild(truncated)
    difficult = doc.createElement('difficult')
    difficult.appendChild(doc.createTextNode(str(0)))
    obj.appendChild(difficult)
    bndbox = doc.createElement('bndbox')

    left = bbx[1]
    top  = bbx[2]
    right = bbx[3]
    bottom = bbx[4]

    xmin = doc.createElement('xmin')
    xmin.appendChild(doc.createTextNode(str(int(left))))
    bndbox.appendChild(xmin)    
    ymin = doc.createElement('ymin')                
    ymin.appendChild(doc.createTextNode(str(int(top))))
    bndbox.appendChild(ymin)                
    xmax = doc.createElement('xmax')                
    xmax.appendChild(doc.createTextNode(str(int(right))))
    bndbox.appendChild(xmax)                
    ymax = doc.createElement('ymax')    
    ymax.appendChild(doc.createTextNode(str(int(bottom))))
    bndbox.appendChild(ymax)

    obj.appendChild(bndbox)                
    return obj

# xml file name: file.xml eg 000013.xml
def create(xmlRootPath, filename, width1, height1, depth1, mat):
    for objIndex, bbx in enumerate(mat):
        if objIndex==0:
            # generate head info of xml file
            filenameString = filename
            folderString = '/export/home/tsq/ssd.pytorch/data/BAIDU/images'

            doc = Document()
            annotation = doc.createElement('annotation')
            doc.appendChild(annotation)
            
            folder = doc.createElement('folder')
            folder.appendChild(doc.createTextNode(folderString))
            annotation.appendChild(folder)
            
            filename = doc.createElement('filename')
            filename.appendChild(doc.createTextNode(filenameString))
            annotation.appendChild(filename)
            
            source = doc.createElement('source')                
            database = doc.createElement('database')
            database.appendChild(doc.createTextNode('LIP FashionDataset'))
            source.appendChild(database)
            source_annotation = doc.createElement('annotation')
            source_annotation.appendChild(doc.createTextNode('LIP FashionDataset'))
            source.appendChild(source_annotation)
            image = doc.createElement('image')
            image.appendChild(doc.createTextNode('flickr'))
            source.appendChild(image)
            flickrid = doc.createElement('flickrid')
            flickrid.appendChild(doc.createTextNode('NULL'))
            source.appendChild(flickrid)
            annotation.appendChild(source)
            
            owner = doc.createElement('owner')
            flickrid = doc.createElement('flickrid')
            flickrid.appendChild(doc.createTextNode('NULL'))
            owner.appendChild(flickrid)
            name = doc.createElement('name')
            name.appendChild(doc.createTextNode('SYSU and CMU'))
            owner.appendChild(name)
            annotation.appendChild(owner)
            
            size = doc.createElement('size')
            width = doc.createElement('width')
            width.appendChild(doc.createTextNode(str(width1)))
            size.appendChild(width)
            height = doc.createElement('height')
            height.appendChild(doc.createTextNode(str(height1)))
            size.appendChild(height)
            depth = doc.createElement('depth')
            depth.appendChild(doc.createTextNode(str(depth1)))
            size.appendChild(depth)
            annotation.appendChild(size)
            
            segmented = doc.createElement('segmented')
            segmented.appendChild(doc.createTextNode(str(0)))
            annotation.appendChild(segmented)
            # generate object info
            if isValidBox(bbx, width1, height1):
                annotation.appendChild(insertObject(doc, bbx))
        else:
            # generate object info
            if isValidBox(bbx, width1, height1):
                annotation.appendChild(insertObject(doc, bbx))
    xmlName = xmlRootPath+filenameString.split('.')[0]+'.xml'
    try:
        f = open(xmlName, "w")
        f.write(doc.toprettyxml(indent = '    '))
        f.close()
    except:
        pass


def getAnnos(fidin, num):
    mat = []
    for i in range(num):
        line = fidin.readline().strip('\n')
        line = map(int, line.split(' '))
        mat.append(list(line))
    return mat, fidin

if __name__ == "__main__":

    '''
        File name eg 'JPEGImages/997_1.jpg'
        Bounding box
            ClassId xmin ymin xmax ymax
    '''
    Bbx_train_gt_txt = './train.txt'

    rootPath = '/export/home/tsq/ssd.pytorch/data/BAIDU/images'

    xmlTrainRootPath = './train-xml/'

    img_id2bbx = {}
    for name in os.listdir('./train'):
        img_id2bbx[name] = []
    lines = open('./train.txt').readlines()
    for line in lines:
        line = line.strip()
        name = line.split(' ')[0]
        temp = [int(ii) if '-' not in ii else 0 for ii in line.split(' ')[1:]]
        temp[0] = temp[0]-1
        img_id2bbx[name].append(temp)


    for name in img_id2bbx.keys():
        img = cv2.imread(osp.join('./train/', name))
        height, width, depth = img.shape
        mat = img_id2bbx[name]
        create(xmlTrainRootPath, name, width, height, depth, mat)