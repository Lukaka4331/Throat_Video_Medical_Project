import os, sys
from glob import glob
from my_package.module_convert_a import convert
from tqdm import tqdm
#讀檔建立資料夾
# mp4_files = glob(os.path.join(r"D:/Medical_Imaging_Project/video0929/Group 1_ RLN/", "*.mp4"))
# file="D:/Medical_Imaging_Project/video0929/Group 1_ RLN/"
# mp4_files = glob(os.path.join(r"D:/Medical_Imaging_Project/video0929/Group 2_ RSLN/", "*.mp4"))
# file="D:/Medical_Imaging_Project/video0929/Group 2_ RSLN/"
mp4_files = glob(os.path.join(r"D:/Medical_Imaging_Project/video0929/Group 3_ Typical speakers/", "*.mp4"))
file="D:/Medical_Imaging_Project/video0929/Group 3_ Typical speakers/"
file_list=os.listdir(file)
print(file_list)

for number in range(len(file_list)):
#     print(mp4_files[mp4_file])
    basename = os.path.basename(file_list[number]) # basename - example.py
    filename = os.path.splitext(basename)[0]  # filename - example 
    # folder="D:/Medical_Imaging_Project/video0929/Group 1_ RLN/"+filename    
    # folder="D:/Medical_Imaging_Project/video0929/Group 2_ RSLN/"+filename
    folder="D:/Medical_Imaging_Project/video0929/Group 3_ Typical speakers/"+filename
    os.mkdir(folder)#創建目錄
    # print(filename)
    print(folder)
    convert(mp4_files[number],folder)