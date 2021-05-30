# model_train.py
* 主要模型訓練-resnet50+fpn
![model](https://user-images.githubusercontent.com/22143034/120103296-86b6c900-c181-11eb-8db9-cae3112f08b8.png)

# predict_mask_output

* Python 中 Shell Script 的使用方法
    * sh 後面接 .sh 文件的路徑
```
sh shell_name
```
## Python 超好用標準函式庫 argparse
resource
* [Python 超好用標準函式庫 argparse](https://dboyliao.medium.com/python-%E8%B6%85%E5%A5%BD%E7%94%A8%E6%A8%99%E6%BA%96%E5%87%BD%E5%BC%8F%E5%BA%AB-argparse-4eab2e9dcc69)

# eval_data.py

* 評估結果
* 預測各類別分數

到pycocotools下修改cocoeval.py

```
def summarize(self):
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
    
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])    
#---------------------------下方修改-----------------------    
                #cacluate AP(average precision) for each category
                num_classes = 3 #這邊是真實類別數(並沒有加入背景)
                avg_ap = 0.0
                if ap == 1:
                    for i in range(0, num_classes):
                        print('category : {0} : {1}'.format(i+1,np.mean(s[:,:,i,:])))
                        avg_ap +=np.mean(s[:,:,i,:])
                    print('(all categories) mAP : {}'.format(avg_ap / num_classes))
    
```

* 預計結果

![image](https://user-images.githubusercontent.com/22143034/120103408-0c3a7900-c182-11eb-8c7d-ae65e370f7c5.png)
