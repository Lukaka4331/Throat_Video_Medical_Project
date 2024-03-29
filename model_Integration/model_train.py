
import os
import numpy as np
import torch
from PIL import Image
import random

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
from torchvision.transforms import functional as F


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
        print(img_path)
        mask_path = os.path.join(self.root, "label", self.masks[idx])
        
        img = Image.open(img_path).convert("RGB")
        img = F.rotate(img,random.randrange(1, 15),fill=0)

        png_file_image=img_path.lstrip("D:/deep_learning_resource/mask_rcnn/filter_data/train_val/image\ ")

        # png_file_image=img_path.lstrip("train_val\image\ ")
        path_image = "D:/deep_learning_resource/mask_rcnn/filter_data/new_rotation_test_val/image"#資夾名稱


        img.save("%s/rr_%s" % (path_image,png_file_image))

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        # mask=mask.rotate(15)
        mask=F.rotate(mask,random.randrange(1, 15),fill=0)

        png_file_label=img_path.lstrip("D:/deep_learning_resource/mask_rcnn/filter_data/train_val/label\ ")
        
        # png_file_label=mask_path.lstrip("train_val\label \ ")
        path_label = "D:/deep_learning_resource/mask_rcnn/filter_data/new_rotation_test_val/label"#資夾名稱


        mask.save("%s/rr_%s" % (path_label,png_file_label))

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # print(type(masks))
        # print(mask)

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            # print(masks[i])
            pos = np.where(masks[i])
            # print(pos)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            # print(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #原本的    

        # there is only one class
#         labels = torch.ones((num_objs,), dtype=torch.int64)
        #  Multi Class
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # print(masks)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # print(area)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        # print(target["boxes"])
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

class Dataset(object):
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
        print(img_path)
        mask_path = os.path.join(self.root, "label", self.masks[idx])
        
        img = Image.open(img_path).convert("RGB")

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        # mask=mask.rotate(15)

        



        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # print(type(masks))
        # print(mask)

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            # print(masks[i])
            pos = np.where(masks[i])
            # print(pos)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            # print(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #原本的    

        # there is only one class
#         labels = torch.ones((num_objs,), dtype=torch.int64)
        #  Multi Class
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # print(masks)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # print(area)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        # print(target["boxes"])
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


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)


    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    # transforms = []
    # transforms.append(T.ToTensor())
    if train:
        print("training")
        # transforms.append(T.RandomHorizontalFlip(0.5))

        transforms = []
        transforms.append(T.ToTensor())        
        # transforms.append(T.RandomRotation())


        # transforms.append(T.ToTensor())        

#         # train_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])

# #         transforms.append(torchvision.transforms.functional.rotate(image,angle=15))
    else :
        transforms = []
        transforms.append(T.ToTensor())

    return T.Compose(transforms)


def main():
    # os.environ['CUDA_VISIBLE_DEVICES']="0"
    # os.environ['CUDA_LAUNCH_BLOCKING'] ="1"
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # CUDA_LAUNCH_BLOCKING=1
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 4
    # use our dataset and defined transformations
    dataset = PennFudanDataset('train_val', get_transform(train=True))
    dataset_test = Dataset('test', get_transform(train=False))


    # split the dataset in train and test set
#     torch.manual_seed(1)
#     indices = torch.randperm(len(dataset)).tolist()
#     dataset = torch.utils.data.Subset(dataset, indices[:-50])
#     dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 1

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    #model_save
#     torch.save({'model': model.state_dict(),
#             },  os.path.join('/maskrcnn-benchmark/test_model/model_weight', 'model_0208_10_vaild_orignaldata.pth'))
#         if epoch % 50 == 0:
        torch.save({'model': model.state_dict(),
            },  os.path.join('model_weight',
                            'model_0527_train_0{}.pth'.format(str(epoch))))
    
#     if epoch % 50 == 0:
#    torch.save(net.state_dict(),'%d.pth' % (epoch))
    print("That's it!")
    
if __name__ == "__main__":
    main()





