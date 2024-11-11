import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import torchvision.transforms as transforms
from torchvision import datasets
import json

class MNIST_CFAR10Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        # self.image_files = sorted([f for f in self.image_dir.iterdir() if f.name.startswith('image_')])
        self.labels = pd.read_csv(self.image_dir/'labels.csv')
        # sort self.labels based on column 'image_id'
        self.labels = self.labels.sort_values(by='image_id').set_index('image_id')
        self.image_ids = self.labels.index.values
        

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = Image.open(self.image_dir/f'image_{image_id}.png').convert('RGB')
        label = (self.labels.loc[image_id]['mnist_label'], self.labels.loc[image_id]['cifar10_label'])  

        if self.transform:
            original_image = transforms.ToTensor()(image)
            image_aug1 = self.transform(image)
            image_aug2 = self.transform(image)
            return original_image, image_aug1, image_aug2, label
        else:
            image = transforms.ToTensor()(image)
            return image, image, image, label
        

class TwoAugImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(TwoAugImageFolder, self).__init__(root, transform)
        self.original_image_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)

        if self.transform is not None:
            original_image = self.original_image_transforms(image)
            image_aug1 = self.transform(image)
            image_aug2 = self.transform(image)
        else:
            original_image = self.original_image_transforms(image)
            image_aug1 = original_image
            image_aug2 = original_image

        return original_image, image_aug1, image_aug2, target
    

class ImageNetValDataset(datasets.VisionDataset):
    def __init__(self, root, transform=None):
        super(ImageNetValDataset, self).__init__(root, transform=transform)
        root = Path(root)
        self.transform = transform
        self.original_image_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        ])

        # Read the validation labels file
        val_labels_mapping = Path('datasets')/'wnid_to_ids.json'
        if not val_labels_mapping.exists():
            raise FileNotFoundError(f"Validation labels file not found at {val_labels_mapping}")
        
        wnid_to_ids = json.load(open(val_labels_mapping))
        
        self.samples = []
        for image in root.iterdir():
            if image.name.endswith('.JPEG'):
                wnid = image.name.rstrip('.JPEG').split('_')[-1]
                self.samples.append((image, wnid_to_ids[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            original_image = self.original_image_transforms(image)
            image_aug1 = self.transform(image)
            image_aug2 = self.transform(image)
        else:
            original_image = self.original_image_transforms(image)
            image_aug1 = original_image
            image_aug2 = original_image

        return original_image, image_aug1, image_aug2, target