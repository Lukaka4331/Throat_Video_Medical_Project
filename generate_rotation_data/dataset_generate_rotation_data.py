
import os
import numpy as np
import torch
from PIL import Image
import random

# import torchvision
from torchvision.transforms import functional as F
from tqdm import tqdm

class RotationDataset(object):
    def __init__(self, root):
        self.root = root
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "image"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "label"))))

    def __getitem__(self, idx):
        # load images ad masks
        # img_path = os.path.join(self.root, "image", self.imgs[idx])
        img_path = os.path.join(self.root, "image", self.imgs[idx])

        # print(img_path)
        # mask_path = os.path.join(self.root, "label", self.masks[idx])
        mask_path = os.path.join(self.root, "label", self.masks[idx])
        # print(mask_path)

        angle=random.randrange(-20, 20)
        img = Image.open(img_path).convert("RGB")
        img = F.rotate(img,angle,fill=0)

        png_file_image=img_path.lstrip("D:/deep_learning_resource/mask_rcnn/filter_data/train_val/image\ ")

        path_image = "D:/deep_learning_resource/mask_rcnn/filter_data/new_rotation_test_val/image03"#資夾名稱03
        img.save("%s/rrrr_%s" % (path_image,png_file_image))#資夾名稱03



        mask = Image.open(mask_path)
        mask=F.rotate(mask,angle,fill=0)
        png_file_label=img_path.lstrip("D:/deep_learning_resource/mask_rcnn/filter_data/train_val/label\ ")

        print(png_file_label)

        path_label = "D:/deep_learning_resource/mask_rcnn/filter_data/new_rotation_test_val/label03"#資夾名稱02
        mask.save("%s/rrrr_%s" % (path_label,png_file_label))#資夾名稱03


        return img

    def __len__(self):
        return len(self.imgs)


def main():

    # use our dataset 
    path_image = "D:/deep_learning_resource/mask_rcnn/filter_data/new_rotation_test_val/image03"#資夾名稱03

    if not os.path.exists(path_image):
        print('mkdir ' + path_image)
        os.mkdir(path_image)   

    path_label = "D:/deep_learning_resource/mask_rcnn/filter_data/new_rotation_test_val/label03"#資夾名稱02

    if not os.path.exists(path_label):
        print('mkdir ' + path_label)
        os.mkdir(path_label)   

    dataset = RotationDataset('train_val')

    # 'D:\deep_learning_resource'

    print('generate_data')
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        # print("rotation_processing")

    print("That's it!")
    
if __name__ == "__main__":
    main()





