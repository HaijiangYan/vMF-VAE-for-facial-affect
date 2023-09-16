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
    def __init__(self, images, labels, labels_identity, input_size, base_transform, aug_transform, augment=False):
        self.images = images
        self.labels = labels
        self.labels_identity = labels_identity
        self.base_transform = base_transform
        self.aug_transform = aug_transform
        self.input_size = input_size
        self.augment = augment

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

        if self.augment:
            base_img = self.base_transform(img)
            aug_img = self.aug_transform(img)
            label = torch.tensor(self.labels[idx]).type(torch.long)
            label_identity = torch.tensor(self.labels_identity[idx]).type(torch.long)

            return base_img, aug_img, label, label_identity

        elif not self.augment:
            base_img = self.base_transform(img)
            label = torch.tensor(self.labels[idx]).type(torch.long)
            label_identity = torch.tensor(self.labels_identity[idx]).type(torch.long)

            return base_img, label, label_identity

        else:
            raise NotImplemented


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
    labels_identity = read_label(os.path.join(path, 'label_identity.csv'))
    # read labels from the csv where labels have already been ordered.

    return filenames, labels, labels_identity


def get_dataloaders(path='./data/cafe/balance_all', bs=64, augment=False, input_size=(64, 40)):
    """ Prepare dataloaders
        Augment training data using:
            - horizental flipping
            - rotation
        input: path to ft_dataset file
        output: Dataloaders """

    filenames, labels, labels_identity = load_data(path)

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,), std=(1,))
    ])

    if augment:
        aug_transform = transforms.Compose([
            # transforms.Grayscale(),
            # transforms.RandomResizedCrop((75, 55), scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            # transforms.RandomApply(
            #     [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
            # transforms.FiveCrop((64, 40)),
            # transforms.Lambda(lambda crops: torch.stack(
            #     [transforms.ToTensor()(crop) for crop in crops])),
            # transforms.Lambda(lambda tensors: torch.stack(
            #     [transforms.Normalize(mean=(0,), std=(1,))(t) for t in tensors])),
            # transforms.Lambda(lambda tensors: torch.stack(
            #     [transforms.RandomErasing()(t) for t in tensors])),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0,), std=(1,))
        ])
    else:
        aug_transform = base_transform

    dataset = CustomDataset(filenames, labels, labels_identity, input_size, base_transform, aug_transform, augment)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=2)
    # use multi-thread to load the dataset

    return dataloader
