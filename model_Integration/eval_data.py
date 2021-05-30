import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "image"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "label"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "image", self.imgs[idx])
        mask_path = os.path.join(self.root, "label", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
#         labels = torch.ones((num_objs,), dtype=torch.int64)
        #  Multi Class
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES']="0"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 4
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,num_classes=num_classes)
    model.to(device)
    model.eval()
    # save = torch.load('D:/deep_learning_resource/mask_rcnn/filter_data/model_weight/model_0503_train_09.pth')
    save = torch.load('D:/deep_learning_resource/mask_rcnn/filter_data/model_weight/model_0530_augument_rotation_final_09.pth')

    model.load_state_dict(save['model'])

    # dataset_test_mix = PennFudanDataset('test_mix', get_transform(train=False))
    
    # data_loader_test_mix = torch.utils.data.DataLoader(
    #     dataset_test_mix, batch_size=1, shuffle=False, num_workers=0,
    #     collate_fn=utils.collate_fn)
    
    
    
    dataset_test= PennFudanDataset('test', get_transform(train=False))
    # dataset_test= PennFudanDataset('0526_test01', get_transform(train=False))

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=3, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)
    
    print('test')
    evaluate(model, data_loader_test, device=device)

    
    
