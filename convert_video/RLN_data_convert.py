import os, sys
from glob import glob
from my_package.module_convert_a import convert
from tqdm import tqdm
#讀檔建立資料夾
mp4_files = glob(os.path.join(r"D:/Medical_Imaging_Project/RLN_183/video", "*.mp4"))
path = "D:/Medical_Imaging_Project/RLN_183/RLN_video_data_{}"#資夾名稱D:\Medical_Imaging_Project\RLN_183
i=0#計算資料夾編號
print(len(mp4_files))

for mp4_file in tqdm(mp4_files):
    print(mp4_file)
# 創建目錄

    i+=1#資料夾編號
    #print(i)
    folder=path.format(i)#串接資料夾編號名字
    #os.mkdir(folder, 755 )#創建目錄
    os.mkdir(folder)#創建目錄
    
    convert(mp4_file,folder)
    print("目錄創建_ok"+str(i))
    print("convert_ok"+str(i))
print("目錄創建_ok_ALL")
print("convert_ok_ALL")
