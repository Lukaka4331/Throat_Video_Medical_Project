import cv2
'''
#convert function()把影片轉為圖片
def convert(folder,file,output_path):
    #使用OpenCV輸入mp4影片：
    vidcap = cv2.VideoCapture(folder + file)
    #要得到影片中的下一幀，我們需要使用：vidcap.read()
    #It returns two values:
        #success: true/ false 
            # true 表示成功讀入
        #image: 圖片資料 如果success==false的話，回傳false    
    success, image = vidcap.read()
    #計算數量
    count = 1
    #用while迴圈判斷
    while success:
        #使用OpenCV的imwrite (func)存為jpg圖片
        cv2.imwrite("%s/image_%d.jpg" % (output_path, count), image)    
        #要讀取所有圖片，我們需要重複調用vidcap.read（），直到它返回false。(沒圖片的時候)
        success, image = vidcap.read()
        print('Saved image ', count)
        #加1為下一筆
        count += 1
    print('ok')   


'''
#from tqdm import tqdm
#convert function()把影片轉為圖片
def convert(file,output_path):
    #使用OpenCV輸入mp4影片：
    vidcap = cv2.VideoCapture(file)
    #要得到影片中的下一幀，我們需要使用：vidcap.read()
    #It returns two values:
        #success: true/ false 
            # true 表示成功讀入
        #image: 圖片資料 如果success==false的話，回傳false    
    success, image = vidcap.read()
    #計算數量
    count = 1
    #用while迴圈判斷
    while success:
        #使用OpenCV的imwrite (func)存為jpg圖片
        cv2.imwrite("%s/image_%d.jpg" % (output_path, count), image)    
        #要讀取所有圖片，我們需要重複調用vidcap.read（），直到它返回false。(沒圖片的時候)
        success, image = vidcap.read()
        #print('Saved image ', count)
        #加1為下一筆
        count += 1
    print('ok')   