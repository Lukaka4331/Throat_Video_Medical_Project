#-*- coding: UTF-8 -*-
from PIL import Image
from PIL import ImageFilter
import os, sys

# def image_filters_test():
#   im = Image.open("D:\Medical_Imaging_Project\RSLN_45\RSLN_video_data_5\image_334.jpg")
#   #im = Image.open(file)

#   #预定义的图像增强滤波器
#   im_min = im.filter(ImageFilter.GaussianBlur)
#   #im.show()
#   im_min.show()
#   #im_min.save('D:/Medical_Imaging_Project/RSLN_45/RSLN_video_data_5_stripe/test.jpg')
#   #cv2.imwrite("%s/image_%d.jpg" % (output_path, count), image)
#   return
# image_filters_test()


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
'''
# 读取图片
img = cv.imread("D:\Medical_Imaging_Project\RSLN_45\RSLN_video_data_5\image_131.jpg", cv.IMREAD_UNCHANGED)
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

kernel = np.ones((5,5),np.float32)/25

dst = cv.filter2D(rgb_img, -1, kernel)

titles = ['Source Image', 'filter2D Image']
images = [rgb_img, dst]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

'''

import os, sys
from glob import glob
from tqdm import tqdm
import re

#讀檔建立資料夾
jpg_files = glob(os.path.join(r"D:/Medical_Imaging_Project/match_data/RSLN_data/orginal/RSLN_data_24_M_R_412964", "*.jpg"))
path = "D:/Medical_Imaging_Project/match_data/RSLN_data/RSLN_data_24_M_R_412964"#資夾名稱
os.mkdir(path)
#去除條紋-remove_stripe
def image_filters(file,count_name):
    im = Image.open(file)
    im_min = im.filter(ImageFilter.MinFilter(3))
    im_min.save("%s/image_%d.jpg" % (path,int(count_name)))

i=0#計算資料夾編號

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

