# Throat_Video_Medical_Project

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
```python=
conda env create -f ../Throat_Video_Medical_Project/environment.yml
```
