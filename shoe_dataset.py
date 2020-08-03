import os
import torch
import torchvision
from skimage import io
from PIL import Image
import numpy as np



from torch.utils.data import DataLoader, Dataset


class ShoeDataSet(Dataset):

    def __init__(self, data_df, transform = None):

        self.data_df = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()


        img_name = self.data_df.iloc[idx, 0]

        image = Image.open(img_name).convert('RGB')

        arr_img = np.array(image)

        label = self.data_df.iloc[idx, 1]


        sample  ={"image":image, "label":label}


        if self.transform:
            image,label = sample["image"],sample["label"]
            image = self.transform(image)
        image = np.array(image)

        sample = {"image":image, "label":label, "file_name":img_name}

        return sample


def train_transformations():

    transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomPerspective(p = 0.4), # randomly change img perspective
        torchvision.transforms.RandomHorizontalFlip(p = 0.2),
        torchvision.transforms.Resize((300,300)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5,),(0.5,0.5,0.5)),
    ])

    return transforms_train


def valid_transformations():

    transforms_valid = torchvision.transforms.Compose([
        torchvision.transforms.Resize((300,300)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5,),(0.5,0.5,0.5))
    ])

    return transforms_valid