# Throat_Video_Medical_Project

簡介

* 預測左右聲帶及聲門，完整segment出各自的形狀

分別為閉合跟打開的狀態
* 閉合
![10_10](https://user-images.githubusercontent.com/22143034/120104655-1d868400-c188-11eb-9bff-159d058389b1.png)
* 打開
![10_103](https://user-images.githubusercontent.com/22143034/120104679-3d1dac80-c188-11eb-837b-6f99c098cee9.png)


Convert mp4 -> jpg

* 影片是圖像的集合。我們可以把視頻當作文本數據，然後圖像是字符。
* 從視頻中提取圖像（也稱為幀）對於各種用例來說都很重要，例如圖像處理，詳細分析影片的一部分，影片編輯等等。
* 1幀約為30張圖像

在本文中，我們將研究如何從視頻中提取圖像並將它們依次保存在資料夾中。我們將使用Python作為編程語言，並將使用OpenCV Library從影片中提取圖像。

安裝OpenCV
```python=
pip install opencv-python
```

環境
```python=
conda install pytorch==1.5.0 torchvision==0.6.0 -c pytorch
```

匯入 environment.yml
* 就能得到 這個環境底下 透過 conda 以及 pip 安裝的套件

```python=
conda env create -f ../Throat_Video_Medical_Project/environment.yml
```
