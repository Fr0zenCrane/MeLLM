import os
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import utils.transforms as OFA_T 
from PIL import Image
import cv2 as cv
import time
import json

class COCODataset(Dataset):
    def __init__(self, args, split="val", tokenize=None, verbose=True):

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

        self.normalize = transforms.Normalize(mean, std)

        raw_data = json.load(open("./data/coco_captions/dataset_coco.json"))['images']
        self.data = []
        for item in raw_data:
            if item['split'] == "restval":
                img_id = item['imgid']
                img_path = os.path.join('./data/VQAv2/', item['filepath'], item['filename'])
                sentences = []
                for sentence in item['sentences']:
                    sentences.append(sentence['raw'])
                self.data.append({'img_id':img_id, 'img_path':img_path, 'captions':sentences})
        self.prompts = [
                    "Describle the given image in a sentence.",
                    "What does this image illustrate?",
                    "What does the image show?",
                    "Please caption this image for me."
                    ]
    def __getitem__(self, index):
        # image input
        item = self.data[index]
        
        img_id = item['img_id']
        image_path = item['img_path']
        captions = item['captions']
        instruction = random.choice(self.prompts)
        img_size = Image.open(image_path).size
        img = cv.imread(image_path)
        assert self.args.encoder_type in ['OFA', 'beit3']
        if self.args.encoder_type == 'OFA':
            img = cv.resize(img, (480, 480))[:, :, ::-1]
        elif self.args.encoder_type == 'beit3':
            img = cv.resize(img, (224, 224))[:, :, ::-1]
        img = self.image_transforms(img.copy())

        sample = {'img_id':img_id, 'img':img, 'instruction':'Describe the image in a sentence', \
                  'input':"", 'captions':captions}
      
        return sample

    def __len__(self):
        return len(self.data)


class CaptionDataset(Dataset):
    def __init__(self, args, split="val", tokenize=None, verbose=True):

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

        self.normalize = transforms.Normalize(mean, std)

        self.data = json.load(open("./data/caption_data_test.json"))

    def __getitem__(self, index):
        # image input
        item = self.data[index]
        img_path = os.path.join('./data/cc_sbu_align/image', item['img_path'])
        caption = item['caption']
        instruction = 'Describle the given image in detail.'

        img = cv.imread(img_path)
        assert self.args.encoder_type in ['OFA', 'beit3']
        if self.args.encoder_type == 'OFA':
            img = cv.resize(img, (480, 480))[:, :, ::-1]
        elif self.args.encoder_type == 'beit3':
            img = cv.resize(img, (224, 224))[:, :, ::-1]
        img = self.image_transforms(img.copy())

        sample = {'img':img, 'instruction':instruction, \
                  'input':"", 'caption':caption}
      
        return sample

    def __len__(self):
        return len(self.data)
