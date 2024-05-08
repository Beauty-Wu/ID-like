import sys
import os
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from transformers import CLIPModel
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from dataloaders import StanfordCars, Food101, OxfordIIITPet, Cub2011
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from tqdm import tqdm
import config
from clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def set_train_loader(args, subset=False, max_count=0):
    root = args.root_dir
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    batch_size = args.batch_size
    batch_size = 256
    shuffle = True
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
    elif args.in_dataset == "ImageNet100":
        path = os.path.join(root, "ImageNet100", 'train')
    elif args.in_dataset == "ImageNet10":
        path = os.path.join(root, "ImageNet10", 'train')
    elif args.in_dataset == "ImageNet20":
        path = os.path.join(root, "ImageNet20", 'train')
    dataset = datasets.ImageFolder(path, transform=preprocess)
    if subset:
        from collections import defaultdict
        classwise_count = defaultdict(int)
        indices = []
        for i, label in enumerate(dataset.targets):
            if classwise_count[label] < max_count:
                indices.append(i)
                classwise_count[label] += 1
        dataset = torch.utils.data.Subset(dataset, indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    return train_loader


def set_val_loader(args):
    root = args.root_dir
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'val')
        dataset = datasets.ImageFolder(path, transform=preprocess)
    elif args.in_dataset == "ImageNet100":
        path = os.path.join(root, "ImageNet100", 'val')
        dataset = datasets.ImageFolder(path, transform=preprocess)
    elif args.in_dataset == "ImageNet10":
        path = os.path.join(root, "ImageNet10", 'train')
        dataset = datasets.ImageFolder(path, transform=preprocess)
    elif args.in_dataset == "ImageNet20":
        path = os.path.join(root, "ImageNet20", 'train')
        dataset = datasets.ImageFolder(path, transform=preprocess)
    elif args.in_dataset == "car196":
        path = root
        dataset = StanfordCars(path, split="test", download=True, transform=preprocess)
    elif args.in_dataset == "food101":
        path = root
        dataset = Food101(path, split="test", download=True, transform=preprocess)
    elif args.in_dataset == "pet37":
        path = root
        dataset = OxfordIIITPet(path, split="test", download=True, transform=preprocess)
    elif args.in_dataset == "bird200":
        path = root
        dataset = Cub2011(path, train=False, transform=preprocess)
    elif args.in_dataset == "cifar10":
        path = root
        dataset = CIFAR10(path, train=False, transform=preprocess)
    elif args.in_dataset == "cifar100":
        path = root
        dataset = CIFAR100(path, train=False, transform=preprocess)

    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return val_loader


def set_ood_loader_ImageNet(args, out_dataset):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    root = os.path.join(args.root_dir, 'ImageNet_OOD_dataset')
    # normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365':  # filtered places
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Places'), transform=preprocess)
    elif out_dataset == 'placesbg':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'placesbg'), transform=preprocess)
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'), transform=preprocess)
    elif out_dataset == 'svhn':
        testsetout = SVHN(root=os.path.join(args.root_dir, 'svhn'), split='test', transform=preprocess)
    elif out_dataset == "cifar10":
        testsetout = CIFAR10(root=args.root_dir, train=False, transform=preprocess)
    elif out_dataset == "cifar100":
        testsetout = CIFAR100(root=args.root_dir, train=False, transform=preprocess)
    elif out_dataset == 'ssb_hard':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(args.root_dir, 'ssb_hard'), transform=preprocess)
    elif out_dataset == 'ninco':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(args.root_dir, 'ninco'), transform=preprocess)
    elif out_dataset == 'openimage_o':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(args.root_dir, 'openimage_o'), transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    return testloaderOut


class RandomCrop(object):
    def __init__(self, n_crop=2):
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        self.n_crop = n_crop
        self.random_crop = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        views = [self.random_crop(x).unsqueeze(dim=0) for _ in range(self.n_crop)]
        views = torch.cat(views, dim=0)
        return views


def set_few_shot_loader(args):
    root = args.root_dir
    data_transform = RandomCrop(args.n_crop)
    # data_transform = RandomCropAndMask(args.n_crop, args.n_crop)
    shuffle = True
    kwargs = {'num_workers': 0, 'pin_memory': True}

    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
        dataset = datasets.ImageFolder(path)
    elif args.in_dataset == "ImageNet100":
        path = os.path.join(root, "ImageNet100", 'train')
        dataset = datasets.ImageFolder(path)
    elif args.in_dataset == "ImageNet10":
        path = os.path.join(root, "ImageNet10", 'train')
        dataset = datasets.ImageFolder(path)
    elif args.in_dataset == "ImageNet20":
        path = os.path.join(root, "ImageNet20", 'train')
        dataset = datasets.ImageFolder(path)
    elif args.in_dataset == "car196":
        path = root
        dataset = StanfordCars(path, split="train", download=True)
        dataset.targets = [target for _, target in dataset]
    elif args.in_dataset == "food101":
        path = root
        dataset = Food101(path, split="train", download=True)
        dataset.targets = [target for _, target in dataset]
    elif args.in_dataset == "pet37":
        path = root
        dataset = OxfordIIITPet(path, split="trainval", download=True)
        dataset.targets = [target for _, target in dataset]
    elif args.in_dataset == "bird200":
        path = root
        dataset = Cub2011(path, train=True)
        dataset.targets = [dataset.data.iloc[idx].target - 1 for idx in range(len(dataset))]
    elif args.in_dataset == "cifar10":
        path = root
        dataset = CIFAR10(path, train=True)
    elif args.in_dataset == "cifar100":
        path = root
        dataset = CIFAR100(path, train=True)

    indices = []
    from collections import defaultdict
    classwise_idx = defaultdict(list)
    print('get dataset index')
    for i, target in enumerate(tqdm(dataset.targets)):
        classwise_idx[target].append(i)
    print('sample few shot dataset')
    from random import sample
    for i in tqdm(range(args.n_cls)):
        sl = sample(classwise_idx[i], args.n_shot)
        indices.extend(sl)

    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
        dataset = datasets.ImageFolder(path, transform=data_transform)
    elif args.in_dataset == "ImageNet100":
        path = os.path.join(root, "ImageNet100", 'train')
        dataset = datasets.ImageFolder(path, transform=data_transform)
    elif args.in_dataset == "ImageNet10":
        path = os.path.join(root, "ImageNet10", 'train')
        dataset = datasets.ImageFolder(path, transform=data_transform)
    elif args.in_dataset == "ImageNet20":
        path = os.path.join(root, "ImageNet20", 'train')
        dataset = datasets.ImageFolder(path, transform=data_transform)
    elif args.in_dataset == "car196":
        path = root
        dataset = StanfordCars(path, split="train", download=True, transform=data_transform)
    elif args.in_dataset == "food101":
        path = root
        dataset = Food101(path, split="train", download=True, transform=data_transform)
    elif args.in_dataset == "pet37":
        path = root
        dataset = OxfordIIITPet(path, split="trainval", download=True, transform=data_transform)
    elif args.in_dataset == "bird200":
        path = root
        dataset = Cub2011(path, train=True, transform=data_transform)
    elif args.in_dataset == "cifar10":
        path = root
        dataset = CIFAR10(path, train=True, transform=data_transform)
    elif args.in_dataset == "cifar100":
        path = root
        dataset = CIFAR100(path, train=True, transform=data_transform)
    dataset = torch.utils.data.Subset(dataset, indices)
    few_shot_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, **kwargs)

    # from torch.utils.data.distributed import DistributedSampler
    # sampler = DistributedSampler(dataset)
    # few_shot_loader = torch.utils.data.DataLoader(dataset, sampler=sampler,
    #                                               batch_size=args.batch_size,
    #                                               shuffle=False, **kwargs)

    return few_shot_loader


def set_few_shot_loader_normal(args):
    root = args.root_dir
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    # data_transform = RandomCropAndMask(args.n_crop, args.n_crop)
    shuffle = True
    kwargs = {'num_workers': 0, 'pin_memory': True}

    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
        dataset = datasets.ImageFolder(path)
    elif args.in_dataset == "ImageNet100":
        path = os.path.join(root, "ImageNet100", 'train')
        dataset = datasets.ImageFolder(path)
    elif args.in_dataset == "ImageNet10":
        path = os.path.join(root, "ImageNet10", 'train')
        dataset = datasets.ImageFolder(path)
    elif args.in_dataset == "ImageNet20":
        path = os.path.join(root, "ImageNet20", 'train')
        dataset = datasets.ImageFolder(path)
    elif args.in_dataset == "car196":
        path = root
        dataset = StanfordCars(path, split="train", download=True)
        dataset.targets = [target for _, target in dataset]
    elif args.in_dataset == "food101":
        path = root
        dataset = Food101(path, split="train", download=True)
        dataset.targets = [target for _, target in dataset]
    elif args.in_dataset == "pet37":
        path = root
        dataset = OxfordIIITPet(path, split="trainval", download=True)
        dataset.targets = [target for _, target in dataset]
    elif args.in_dataset == "bird200":
        path = root
        dataset = Cub2011(path, train=True)
        dataset.targets = [dataset.data.iloc[idx].target - 1 for idx in range(len(dataset))]
    elif args.in_dataset == "cifar10":
        path = root
        dataset = CIFAR10(path, train=True)
    elif args.in_dataset == "cifar100":
        path = root
        dataset = CIFAR100(path, train=True)

    indices = []
    from collections import defaultdict
    classwise_idx = defaultdict(list)
    print('get dataset index')
    for i, target in enumerate(tqdm(dataset.targets)):
        classwise_idx[target].append(i)
    print('sample few shot dataset')
    from random import sample
    for i in tqdm(range(args.n_cls)):
        sl = sample(classwise_idx[i], args.n_shot)
        indices.extend(sl)

    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
        dataset = datasets.ImageFolder(path, transform=data_transform)
    elif args.in_dataset == "ImageNet100":
        path = os.path.join(root, "ImageNet100", 'train')
        dataset = datasets.ImageFolder(path, transform=data_transform)
    elif args.in_dataset == "ImageNet10":
        path = os.path.join(root, "ImageNet10", 'train')
        dataset = datasets.ImageFolder(path, transform=data_transform)
    elif args.in_dataset == "ImageNet20":
        path = os.path.join(root, "ImageNet20", 'train')
        dataset = datasets.ImageFolder(path, transform=data_transform)
    elif args.in_dataset == "car196":
        path = root
        dataset = StanfordCars(path, split="train", download=True, transform=data_transform)
    elif args.in_dataset == "food101":
        path = root
        dataset = Food101(path, split="train", download=True, transform=data_transform)
    elif args.in_dataset == "pet37":
        path = root
        dataset = OxfordIIITPet(path, split="trainval", download=True, transform=data_transform)
    elif args.in_dataset == "bird200":
        path = root
        dataset = Cub2011(path, train=True, transform=data_transform)
    elif args.in_dataset == "cifar10":
        path = root
        dataset = CIFAR10(path, train=True, transform=data_transform)
    elif args.in_dataset == "cifar100":
        path = root
        dataset = CIFAR100(path, train=True, transform=data_transform)
    dataset = torch.utils.data.Subset(dataset, indices)
    few_shot_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, **kwargs)

    # from torch.utils.data.distributed import DistributedSampler
    # sampler = DistributedSampler(dataset)
    # few_shot_loader = torch.utils.data.DataLoader(dataset, sampler=sampler,
    #                                               batch_size=args.batch_size,
    #                                               shuffle=False, **kwargs)

    return few_shot_loader
