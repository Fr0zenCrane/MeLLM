import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2 as cv
import time
import json

IMAGE_PATH_PREFIX = "./data/llava_instruct_150k/train2014/COCO_train2014_"
class PretrainDataset(Dataset):
    def __init__(self, args, split, tokenize=None, verbose=True):

        start_time = time.time()
        self.args = args
        self.split = split
        assert split in ["train", "val"]
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
        
        self.normalize = transforms.Normalize(mean, std)
        if split == 'train':
            self.data = json.load(open("./data/llava_instruct_150k/llava_train_1w.json", mode='r',encoding='utf-8'))
        else:
            self.data = json.load(open("./data/llava_instruct_150k/llava_test.json", mode='r',encoding='utf-8'))

    def __getitem__(self, index):
        # image input
        conversation_id = self.data[index]['id']
        image_path = IMAGE_PATH_PREFIX + self.data[index]['image']
        instruction = self.data[index]['conversations'][0]['value'].replace('\n', '').replace('<image>', '')
        input = ""
        output = self.data[index]['conversations'][1]['value'].replace('\n', '').replace('<image>', '')

        img = cv.imread(image_path)
        if self.args.encoder_type == 'OFA':
            resolution = 480
        elif self.args.encoder_type == 'beit3':
            resolution = 224
        img = cv.resize(img, (resolution, resolution))[:, :, ::-1]
        img = self.image_transforms(img.copy())
        
        return {'img':img, 'conversation_id':conversation_id,
                'instruction':instruction, 'input':input,
                'target':output }

    def __len__(self):
        return len(self.data)
