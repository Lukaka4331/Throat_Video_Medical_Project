# import necessary libraries
# %matplotlib inline
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import os
import cv2
import random
import warnings
from glob import glob
from tqdm import tqdm
import re
import argparse



def main():
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='Model_Predict_Mask')
    parser.add_argument('--predict_image_input', type=str,
                    help='data_input')

    parser.add_argument('--predict_mask_output', type=str,
                    help='mask_output')             

    args = parser.parse_args()
    # set to evaluation mode
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    num_classes = 4
    # load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,num_classes=num_classes)

    model.eval()

    save = torch.load('model_weight/model_0411_train_mix_update_09.pth')
    model.load_state_dict(save['model'])

    

    # load COCO category names

    COCO_CLASS_NAMES = ['background','glottis','throat_left','throat_right']


    
    # jpg_files = glob(os.path.join(r"D:/deep_learning_resource/mask_rcnn/dataset/RSLN_5/image", "*.png"))
    # mask_output = "D:/deep_learning_resource/mask_rcnn/dataset/mask_demo/RSLN_5_demo_mask"#資夾名稱

    #image_input
    # jpg_files = glob(os.path.join(r"{}".format(args.jpg_files), "*.png"))
    # img_input=args.predict_image_input

    # path = args.mkdir_folder

    def get_coloured_mask(mask,pred_cls):

        colours = [[0, 0, 255],[0, 255, 0],[0, 255, 255]]

        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        


        if pred_cls=='glottis' :
            r[mask == 1], g[mask == 1], b[mask == 1] = colours[0]#紅
            coloured_mask = np.stack([r, g, b], axis=2)
        if pred_cls=='throat_left' :
            r[mask == 1], g[mask == 1], b[mask == 1] = colours[1]#綠
            coloured_mask = np.stack([r, g, b], axis=2)
            
        if pred_cls=='throat_right'  :
            r[mask == 1], g[mask == 1], b[mask == 1] = colours[2]#黃
            coloured_mask = np.stack([r, g, b], axis=2)  
        
        return coloured_mask

    def get_prediction(img_path, confidence):

        img = Image.open(img_path)
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        pred = model([img])

        pred_label = list(pred[0]['labels'].detach().numpy())
        print('label: {}'.format(pred_label))
        print('')
        pred_score = list(pred[0]['scores'].detach().numpy())
        print('score: {}'.format(pred_score))
        print('')

        
        pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
        print('Filter -confidence score:{}'.format(pred_t))



        masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()


        pred_class = [COCO_CLASS_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    #     print(pred_class)
        
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        
        
        masks = masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]

        
        return masks, pred_boxes, pred_class

    def segment_instance(img_path, confidence=0.5, rect_th=2, text_size=0.7, text_th=2,outputfile='rln'):

        masks, boxes, pred_cls = get_prediction(img_path, confidence)
        print(pred_cls)
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.zeros(img.shape, np.uint8)# mask a black image of same size with input

        compare_label=[]
        for i,label in zip(range(len(masks)),pred_cls):
    #         print(i,label)

            if label not in compare_label:
                print(label)
                rgb_mask = get_coloured_mask(masks[i],pred_cls[i])
                img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
                cv2.imwrite(outputfile, img)
    #             cv2.imwrite(outputfile+'file_{}.png'.format(i), img)

                compare_label.append(label)
    #             [compare_label.append(i) for i in pred_cls if not i in compare_label]

            else :
                print('have label')
        print(compare_label)    
        


    jpg_files = glob(os.path.join(args.predict_image_input, "*.png"))

    #mask_output
    mask_output = args.predict_mask_output 

    print(os.path.exists(mask_output))

    if not os.path.exists(mask_output):
        print('mkdir ' + mask_output)
        os.mkdir(mask_output)                 

    for jpg_file in tqdm(jpg_files):
        print(jpg_file)
#     print(jpg_files)

        # count0=re.findall("([0-9]+_[0-9]+)", jpg_file)        # RSLN
        # count0=re.findall("([0-9]+_RLN_[0-9]+)", jpg_file)      # RLN
        count0=re.findall("([0-9]+-2693456-20181129_[0-9]+)", jpg_file) #RSLN43-name-different


        print(count0)
        video_id =count0[0]
        print(video_id)
        segment_instance(jpg_file, confidence=0,outputfile='{}/{}.png'.format(mask_output,video_id))



if __name__ == "__main__":
    main()
    
    