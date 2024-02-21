import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2 as cv
import time
import json

IMAGE_PATH_PREFIX = "./cc3m/image/"
class OutDomainDataset(Dataset):
    def __init__(self, args, type='caption', tokenize=None, verbose=True):

        start_time = time.time()
        self.args = args
        self.type = type
        assert type in ["caption", "vqa"]
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
        if self.type == 'caption':
            self.data = json.load(open("./data/out_domain_caption.json", mode='r',encoding='utf-8'))
        else:
            self.data = json.load(open("./data/revised_out_domain_qa.json", mode='r',encoding='utf-8'))

    def __getitem__(self, index):
        # image input
        image_path = IMAGE_PATH_PREFIX + self.data[index]['img']
        instruction = self.data[index]['instruction']
        input = ""
        output = self.data[index]['answer']

        img = cv.imread(image_path)
        if self.args.encoder_type == 'OFA':
            resolution = 480
        elif self.args.encoder_type == 'beit3':
            resolution = 224
        img = cv.resize(img, (resolution, resolution))[:, :, ::-1]
        img = self.image_transforms(img.copy())
        
        return {'img':img, 'img_path':image_path,
                'instruction':instruction, 'input':input,
                'target':output }

    def __len__(self):
        return len(self.data)
