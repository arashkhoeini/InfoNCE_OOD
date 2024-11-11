import os.path as osp
import os
import torch
from torchvision import models
from models.moco2 import MoCo
from models.custom_encoder import CustomEncoder
import random
from PIL import ImageFilter
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.dataset import MNIST_CFAR10Dataset, TwoAugImageFolder, ImageNetValDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from transformers import AutoModelForImageClassification, ResNetModel

def mkdir(p):
    if not osp.exists(p):
        os.makedirs(p)
        print('DIR {} created'.format(p))
    return p

def get_imagenet_transforms():
    # Define your transforms here
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        #transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.RandomErasing(p=0.5)
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transforms, test_transforms

def get_mnist_cifar10_transforms():
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.57580509, 0.56986527, 0.53811235), std=(0.25580952, 0.25461894, 0.26496579))
    ])
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.57580509, 0.56986527, 0.53811235), std=(0.25580952, 0.25461894, 0.26496579))
    ])

    return train_transforms, test_transforms

def get_ood_loader(configs, data_dir):
    ood_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize the shorter side of the image to 256 pixels
        transforms.CenterCrop(224),  # Crop the center 224x224 pixels from the resized image
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet values
    ])

    dataset = ImageFolder(root=data_dir, transform=ood_transforms)
    dataloader = DataLoader(dataset, configs.dataset.batch_size, shuffle=False)
    return dataloader

def get_loaders(configs):
    
    if 'imagenet' in configs.dataset.name:
        loaders = get_imagenet_loaders(configs)
    elif configs.dataset.name == 'mnist_cfar10':
        loaders = get_mnist_cfar10_loaders(configs)
    else:
        raise ValueError("Unknown dataset: {}".format(configs.dataset.name))

    return loaders

def get_imagenet_loaders(configs):
    data_dir = Path(configs.dataset.root)/configs.dataset.name

    train_transforms, test_transforms = get_imagenet_transforms()

    train_dataset = TwoAugImageFolder(data_dir/'train', transform=train_transforms)
    # val_dataset = ImageNetValDataset(data_dir/'val', transform=test_transforms)
    val_dataset = TwoAugImageFolder(data_dir/'val', transform=test_transforms)
    

    if configs.cuda and (configs.world_size > 1):
        train_sampler = DistributedSampler(train_dataset, num_replicas=configs.world_size, rank=configs.rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=configs.world_size, rank=configs.rank)
    else:
        train_sampler = None
        val_sampler = None

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=configs.dataset.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=configs.dataset.batch_size, shuffle=(val_sampler is None), sampler=val_sampler, num_workers=1)
    
    
    loaders = {'train': train_loader, 'val': val_loader, 'test': None}

    return loaders

def get_mnist_cfar10_loaders(configs):
    data_dir = Path(configs.dataset.root)

    train_transforms, test_transforms = get_mnist_cifar10_transforms()

    train_dataset = MNIST_CFAR10Dataset(data_dir/'train', transform=train_transforms)
    val_dataset = MNIST_CFAR10Dataset(data_dir/'val', transform=test_transforms)
    test_dataset = MNIST_CFAR10Dataset(data_dir/'test', transform=test_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=configs.dataset.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.dataset.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=configs.dataset.batch_size, shuffle=False)
    
    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    return loaders

def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

def get_model(configs): 
    if configs.model.method == 'mocov2':
        model = MoCo(models.__dict__[configs.model.encoder], dim=configs.model.feature_dim, K=configs.model.num_negs, T=configs.model.temperature)
    elif configs.model.method in ['supervised_infonce', 'infonce']:
        if configs.model.checkpoint_source == 'microsoft':
            model = CustomEncoder(ResNetModel.from_pretrained("microsoft/resnet-50"), configs.model.feature_dim)
        elif configs.model.checkpoint_source == 'leftthomas':
            model = CustomEncoder(AutoModelForImageClassification.from_pretrained("leftthomas/resnet-50"), configs.model.feature_dim)
        elif configs.model.checkpoint_source == 'francesco':
            model = CustomEncoder(AutoModelForImageClassification.from_pretrained("francesco/resnet-50"), configs.model.feature_dim)
        elif configs.model.checkpoint_source == 'random':
            model = CustomEncoder(models.__dict__[configs.model.encoder](weights=None), configs.model.feature_dim)
        else:
            raise NotImplementedError(f"Checkpoint source {configs.model.checkpoint_source} is unknown!")
    else:
        raise ValueError("Unknown pretraining method: {}".format(configs.pretraining.method))
    
    return model

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def entropy_loss(X, pairwise_distances=None):
    """ 
    Compute the entropy of vector representations (row vectors) in tensor x
    """
    # Compute pairwise distances
    if pairwise_distances is None:
        pairwise_distances = torch.cdist(X, X, p=2)

    probs = torch.nn.functional.softmax(-pairwise_distances, dim=1)
    entropy_values = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)

    return entropy_values.mean()

def recall_at_1(X, labels, pairwise_distances=None):
    """
    Compute the recall@1 for representations stored in tensor X.
    
    Args:
    - X (torch.Tensor): Tensor of shape (n_samples, n_features) containing the representations.
    - labels (torch.Tensor): Tensor of shape (n_samples,) containing the ground truth labels.
    
    Returns:
    - recall (float): The recall@1 value.
    """
    # Compute pairwise distances
    if pairwise_distances is None:
        pairwise_distances = torch.cdist(X, X, p=2)
    
    # Set diagonal to a large value to ignore self-matching
    pairwise_distances.fill_diagonal_(float('inf'))
    
    # Find the nearest neighbor for each sample
    nearest_neighbors = torch.argmin(pairwise_distances, dim=1)
    
    # Compute recall@1
    correct_matches = (labels == labels[nearest_neighbors]).float()
    recall = correct_matches.mean().item()
    
    return recall

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]
    
class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x