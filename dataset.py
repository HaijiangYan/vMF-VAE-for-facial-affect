# dataset cafe import
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2


class CustomDataset(Dataset):
    def __init__(self, images, labels, input_size, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.input_size = input_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.input_size == (64, 40):
            img = Image.fromarray(cv2.imread(self.images[idx], 0))
        elif self.input_size == (40, 40):
            img = Image.fromarray(cv2.imread(self.images[idx], 0)[15:-9, :])
        else: 
            raise NotImplemented
        # imread grayscale picture as array, PIL Image turn it into image
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx]).type(torch.long)

        return img, label


def read_label(filepath):
    """read label.csv and return a list"""

    file = open(filepath, 'r', encoding="utf-8")
    context = file.read()  # as str
    list_result = context.split("\n")[1:-1]
    length = len(list_result)

    for i in range(length):
        list_result[i] = int(list_result[i].split(",")[-1])

    file.close()  # file must be closed after manipulating
    return list_result


def load_data(path):

    filenames = [path + '/' + file for file in os.listdir(path) 
    			if os.path.splitext(file)[-1] == '.jpg']
    filenames.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))  
    # sorted by their order to match the label's order

    labels = read_label(os.path.join(path, 'label.csv'))
    # read labels from the csv where labels have already been ordered.

    return filenames, labels


def get_dataloaders(path='./data/cafe/balance_all', bs=64, augment=False, input_size=(64, 40)):
    """ Prepare dataloaders
        Augment training data using:
            - horizental flipping
            - rotation
        input: path to ft_dataset file
        output: Dataloaders """

    filenames, labels = load_data(path)

    test_transform = transforms.Compose([
        # transforms.Grayscale(),
        # transforms.FiveCrop(40),
        # transforms.Lambda(lambda crops: torch.stack(
        #     [transforms.ToTensor()(crop) for crop in crops])),
        transforms.ToTensor(),
        # transforms.Lambda(lambda tensors: torch.stack(
        #     [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
        transforms.Normalize(mean=(0,), std=(1,))
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.FiveCrop(40),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.RandomErasing()(t) for t in tensors])),
        ])
    else:
        train_transform = test_transform

    dataset = CustomDataset(filenames, labels, input_size, train_transform)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=2)
    # use multi-thread to load the dataset

    return dataloader
