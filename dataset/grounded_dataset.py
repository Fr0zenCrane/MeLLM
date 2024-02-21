import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import utils.transforms as OFA_T 
from PIL import Image
import cv2 as cv
import time
import json
    

def find_useful_obj(text):
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, text)
    return matches


class GroundedQADataset(Dataset):
    def __init__(self, args, split="val", require_encode=False, tokenize=None, verbose=True):

        self.args = args
        self.split = split
        self.require_encode = require_encode
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
        self.data = json.load(open("./data/new_groundedQA.json"))
    def __getitem__(self, index):
        # image input
        item = self.data[index]

        image_path = os.path.join("./data/pmr/", item['img_path'])
        instruction = item['question']
        boxes = item['boxes']
        objects = item['objects']
        print(instruction)
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
            quant_x0 = "<bin_{}>".format(int(((encoded_boxes[i][0] * (self.num_bins - 1) / 512)).round()))
            quant_y0 = "<bin_{}>".format(int(((encoded_boxes[i][1] * (self.num_bins - 1) / 512)).round()))
            quant_x1 = "<bin_{}>".format(int(((encoded_boxes[i][2] * (self.num_bins - 1) / 512)).round()))
            quant_y1 = "<bin_{}>".format(int(((encoded_boxes[i][3] * (self.num_bins - 1) / 512)).round()))
            encoded_boxes[i] = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)   
        premise = ""
        for i, p_item in enumerate(item['premise']):
            if not isinstance(p_item, str):
                premise += ' ' +'['+objects[p_item[0]]+']' #+ encoded_boxes[p_item[0]]
            else:
                premise += ' ' + p_item
        premise = premise.strip()
        obj_prompt = "Useful Object in the image : "
        useful_objects = [objects.index(obj.lower()) for obj in find_useful_obj(item['question'])]
        for a_item in item['answer']:
            if not isinstance(a_item, str):
                obj = objects[a_item[0]]
                if obj not in useful_objects:
                    useful_objects.append(objects.index(obj.lower()))

        obj_prompt = "Useful Object in the image : "
        for i, obj in enumerate(useful_objects):
            if self.require_encode:
                obj_prompt += '[' + objects[obj] + ']' + encoded_boxes[obj] + ' '
            else:
                obj_prompt += '[' + objects[obj] + ']' + \
                     "({} {} {} {})".format(boxes[obj][0], boxes[obj][1], boxes[obj][2], boxes[obj][3]) + ' '
        
        # print(obj_prompt)
        if self.require_encode:
            instruction = obj_prompt + '\n\n\n' + "Considering that: " + premise + '\n\n\n' + instruction
            
        output = ""
        for i, a_item in enumerate(item['answer']):
            if not isinstance(a_item, str):
                output += ' ' + '[' + objects[a_item[0]] + ']'
            else:
                output += ' ' + a_item
        sample = {'img':img, 'img_path':image_path, 'instruction':instruction, 'input':"", 'target':output}
      
        return sample

    def __len__(self):
        return len(self.data)
    
