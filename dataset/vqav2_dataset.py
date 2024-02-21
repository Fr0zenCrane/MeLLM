import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2 as cv
import time
import json


class VQAv2Dataset(Dataset):
    IMAGE_PATH = {
        "train": ("train2014", "v2_OpenEnded_mscoco_train2014_questions.json", "v2_mscoco_train2014_annotations.json"),
        "val": ("val2014", "v2_OpenEnded_mscoco_val2014_questions.json", "v2_mscoco_val2014_annotations.json"),
        "dev": ("val2014", "v2_OpenEnded_mscoco_dev_questions.json", "v2_mscoco_val2014_annotations.json"),
        "testdev": ("test2015", "v2_OpenEnded_mscoco_test-dev2015_questions.json"),
        "test": ("test2015", "v2_OpenEnded_mscoco_test2015_questions.json")}

    def __init__(self, args, split, data_path="", question_transforms=None, tokenize=None,
                 verbose=True, testing=False):
        """
        split train, val, test
        balanced True, False
        image_transforms
        question_transforms
        """
        start_time = time.time()
        self.args = args
        self.split = split
        self.testing = testing
        assert split in ["train", "val", "test-dev", "test", 'dev']
        if split == 'test-dev':
            split='testdev'
        self.data_path = data_path
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
        self.question_transforms = question_transforms
        self.tokenize = tokenize
        path = os.path.expanduser(os.path.join(data_path, self.IMAGE_PATH[split][1]))

        if verbose:
            print(f"Start loading VQAv2 Dataset from {path}", flush=True)

        # Questions
        with open(path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data["questions"])
        df["image_path"] = df["image_id"].apply(
            lambda x: f"{self.IMAGE_PATH[split][0]}/COCO_{self.IMAGE_PATH[split][0]}_{x:012d}.jpg")

        # Annotations
        if not testing:
            path = os.path.expanduser(os.path.join(data_path, self.IMAGE_PATH[split][2]))
            with open(path, 'r') as f:
                data = json.load(f)
            df_annotations = pd.DataFrame(data["annotations"])
            df = pd.merge(df, df_annotations, left_on='question_id', right_on='question_id', how='left')
            df["image_id"] = df["image_id_x"]
            if not all(df["image_id_y"] == df["image_id_x"]):
                print("There is something wrong with image_id")
            del df["image_id_x"]
            del df["image_id_y"]
        self.df = df
        self.n_samples = self.df.shape[0]
        if verbose:
            print(
                f"Loading VQAv2 Dataset done in {time.time() - start_time:.1f} seconds. Loaded {self.n_samples} samples.")

    def __getitem__(self, index):
        # image input
        image_id = self.df.iloc[index]["image_id"]
        image_path = self.df.iloc[index]["image_path"]
        question = self.df.iloc[index]["question"]
        user_prompt = 'Question:' + question +'\n\nAnswer the question in one or two words:'

        question_id = self.df.iloc[index]["question_id"]
        split = self.split
        if not self.testing:
            main_answer = self.df.iloc[index]["multiple_choice_answer"]  # Already extracted main answer
            answers = self.df.iloc[index][
                "answers"]  # list of dicts: [{'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1}, ...]
            full_prompt = user_prompt + main_answer
        # Load and transform image
        image_path = os.path.expanduser(os.path.join(self.data_path, image_path))
        # print("image_path ---->", image_path)
        # with open(image_path, "rb") as f:
        #     img = Image.open(f).convert('RGB')
        # if self.image_transforms:
        #     img = self.image_transforms(img)
        img = cv.imread(image_path)
        if self.args.encoder_type =='OFA':
            img = cv.resize(img, (480, 480))[:, :, ::-1]
        elif self.args.encoder_type == 'beit3':
            img = cv.resize(img, (224, 224))[:, :, ::-1]
        img = self.image_transforms(img.copy())

        # Return
        if self.testing:
            return {"img": img, "image_id": image_id, "question_id": question_id, "question": question,
                    "instruction": user_prompt, 'input':""}
        else:
            return {"img": img, "image_id": image_id, "question_id": question_id, "question": question,
                    "instruction": user_prompt, 'input':"", "target": main_answer}

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.df.shape[0]
