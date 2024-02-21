import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import utils.transforms as OFA_T 
from PIL import Image
import cv2 as cv
import time
import json

INSTRUCT_IMAGE_PATH_PREFIX = "./data/llava_instruct_150k/train2014/COCO_train2014_"
CAPTION_IMAGE_PATH_PREFIX = "./data/cc_sbu_align/image/"
class MixedDataset(Dataset):
    def __init__(self, args, split="train", tokenize=None, verbose=True):

        start_time = time.time()
        self.args = args
        self.split = split
        # assert split in ["train", "val"]
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        #     self.image_transforms = transforms.Compose([
        #     transforms.ToTensor(), 
        #     transforms.Resize((args.resolution, args.resolution), interpolation=Image.BICUBIC),
        #     transforms.Normalize(mean=mean, std=std)
        # ])
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]) 
        self.box_transforms = OFA_T.Compose([
            OFA_T.RandomResize([480], max_size=480),
            OFA_T.ToTensor(),
            OFA_T.Normalize(mean=mean, std=std, max_image_size=480)
        ])
        
        self.normalize = transforms.Normalize(mean, std)
        self.num_bins = 1000
        if self.args.encoder_type != 'beit3':
            self.data = json.load(open("./data/mixed_data.json"))
        else:
            self.data = json.load(open("./data/mixed_data_without_grounded.json"))
    def __getitem__(self, index):
        # image input
        item = self.data[index]
        if item['data_type'] == 'instruction':
            image_path = INSTRUCT_IMAGE_PATH_PREFIX + item['image']
            instruction = item['conversations'][0]['value'].replace('\n', '').replace('<image>', '')
            output = item['conversations'][1]['value'].replace('\n', '').replace('<image>', '')

            img = cv.imread(image_path)
            if self.args.encoder_type == 'OFA':
                resolution = 480
            elif self.args.encoder_type == 'beit3':
                resolution = 224
            img = cv.resize(img, (resolution, resolution))[:, :, ::-1]
            img = self.image_transforms(img.copy())
            sample = {'img':img, 'instruction':instruction, 'input':"", 'target':output}
        elif item['data_type'] == 'caption':
            image_path = CAPTION_IMAGE_PATH_PREFIX + item['img_path']
            instruction = item['instruction']
            input = ""
            output = item['caption']

            img = cv.imread(image_path)
            if self.args.encoder_type == 'OFA':
                resolution = 480
            elif self.args.encoder_type == 'beit3':
                resolution = 224
            img = cv.resize(img, (resolution, resolution))[:, :, ::-1]
            img = self.image_transforms(img.copy())
            sample = {'img':img, 'instruction':instruction, 'input':"", 'target':output}
        elif item['data_type'] == 'vqa':
            image_path = os.path.join("./data/VQAv2/", item['image_path'])
            instruction = item['instruction'] + item['question']
            output = item['answer']

            img = cv.imread(image_path)
            if self.args.encoder_type == 'OFA':
                resolution = 480
            elif self.args.encoder_type == 'beit3':
                resolution = 224
            img = cv.resize(img, (resolution, resolution))[:, :, ::-1]
            img = self.image_transforms(img.copy())
            sample = {'img':img, 'instruction':instruction, 'input':"", 'target':output}
        elif item['data_type'] == 'grounded':
            image_path = os.path.join("./data/pmr/", item['img_path'])
            instruction = item['question']
            boxes = item['boxes']
            encoded_boxes = []
            # encode boxes
            img_size = Image.open(image_path).size
            img = cv.imread(image_path)
            img = cv.resize(img, (480, 480))[:, :, ::-1]
            img = self.image_transforms(img.copy())
            ratio_width, ratio_height = float(img.size(0)) / float(img_size[0]), float(img.size(1)) / float(img_size[1])
            ratio = torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            for box in boxes:
                encoded_box = torch.tensor([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                encoded_box = encoded_box * ratio
                encoded_boxes.append(encoded_box)

            for i, box in enumerate(boxes):
                quant_x0 = "<bin_{}>".format(int((encoded_boxes[i][0] * (self.num_bins - 1)).round()))
                quant_y0 = "<bin_{}>".format(int((encoded_boxes[i][1] * (self.num_bins - 1)).round()))
                quant_x1 = "<bin_{}>".format(int((encoded_boxes[i][2] * (self.num_bins - 1)).round()))
                quant_y1 = "<bin_{}>".format(int((encoded_boxes[i][3] * (self.num_bins - 1)).round()))
                encoded_boxes[i] = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
                # print(encoded_box)
                
            premise = ""
            for i, p_item in enumerate(item['premise']):
                if not isinstance(p_item, str):
                    premise += ' ' + encoded_boxes[p_item[0]]
                else:
                    premise += ' ' + p_item
            premise = premise.strip()
            instruction += "Note: " + premise

            output = item['annotated_answer']

            sample = {'img':img, 'instruction':instruction, 'input':"", 'target':output}
        else:
            print('???')
        
        return sample

    def __len__(self):
        return len(self.data)
    
