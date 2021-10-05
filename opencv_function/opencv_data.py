#-*- coding: UTF-8 -*-
from PIL import Image
from PIL import ImageFilter
import os, sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os, sys
from glob import glob
from tqdm import tqdm
import re

#去除條紋-remove_stripe
def image_filters(file,count_name):
    im = Image.open(file)
    im_min = im.filter(ImageFilter.MinFilter(3))
    im_min.save("%s/image_%d.png" % (path,int(count_name)))

file="D:/Medical_Imaging_Project/video0929/Group 3_ Typical speakers/"
file_list=os.listdir(file)
# print(file_list)
file_name=[]
for item in file_list:
        # print(item)
        if os.path.isdir(file+item):
            # print('資料夾：' + item)
            file_name.append(item)
print(file_name)
# i=0#計算資料夾編號

for i in range(len(file_name)):

    jpg_files = glob(os.path.join(r"D:/Medical_Imaging_Project/video0929/Group 3_ Typical speakers/{}".format(str(file_name[i])), "*.png"))
    path = "D:/Medical_Imaging_Project/video0929/Group 3_ Typical speakers_filter/{}".format(str(file_name[i]))#資夾名稱
    os.mkdir(path)
    for jpg_file in tqdm(jpg_files):

        count=re.findall("image_([0-9]+)", jpg_file)
        count_name =count[0]
        #print(count_name)

    # 創建目錄

        i+=1#資料夾編號

        image_filters(jpg_file,count_name)
        #print("目錄創建_ok"+str(i))

        #print("convert_ok"+str(i))
    print("目錄創建_ok_ALL")
