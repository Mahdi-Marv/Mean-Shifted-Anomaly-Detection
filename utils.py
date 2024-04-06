import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import torchvision.models as models
import torch.nn.functional as F
from PIL import ImageFilter
import random
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from glob import glob

import matplotlib.pyplot as plt


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_resnet18 = transforms.Compose([
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

moco_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class Transform:
    def __init__(self):
        self.moco_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2


class Model(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if backbone == 152:
            self.backbone = models.resnet152(pretrained=True)
        else:
            self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, backbone, train_fc=False)

    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n


def freeze_parameters(model, backbone, train_fc=False):
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
    if backbone == 152:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False
        for p in model.layer2.parameters():
            p.requires_grad = False


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def get_loaders(dataset, label_class, batch_size, backbone):
    if dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10
        transform = transform_color if backbone == 152 else transform_resnet18
        coarse = {}
        trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
        testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        trainset_1 = ds(root='data', train=True, download=True, transform=Transform(), **coarse)
        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        trainset_1.data = trainset_1.data[idx]
        trainset_1.targets = [trainset_1.targets[i] for i, flag in enumerate(idx, 0) if flag]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                                  drop_last=False)
        return train_loader, test_loader, torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                                                      shuffle=True, num_workers=2, drop_last=False)
    else:
        print('Unsupported Dataset')
        exit()


def get_loader_aptos(batch_size, backbone):
    train_normal_path = glob('/kaggle/working/Mean-Shifted-Anomaly-Detection/APTOS1/train/NORMAL/*')
    train_normal_label = [0] * len(train_normal_path)

    test_normal_path = glob('/kaggle/working/Mean-Shifted-Anomaly-Detection/APTOS2/test/NORMAL/*')
    test_anomaly_path = glob('/kaggle/working/Mean-Shifted-Anomaly-Detection/APTOS2/test/ABNORMAL/*')

    test_path = test_normal_path + test_anomaly_path
    test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

    transform = transform_color if backbone == 152 else transform_resnet18

    train_set = APTOS(image_path=train_normal_path, labels=train_normal_label, transform=transform)
    test_set = APTOS(image_path=test_path, labels=test_label, transform=transform)
    train_set1 = APTOS(image_path=train_normal_path, labels=train_normal_label, transform=Transform())

    visualize_random_samples_from_clean_dataset(train_set, "trainset")
    visualize_random_samples_from_clean_dataset(test_set, "testset")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                                               drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=2,
                                              drop_last=False)
    return train_loader, test_loader, torch.utils.data.DataLoader(train_set1, batch_size=batch_size,
                                                                  shuffle=True, num_workers=2, drop_last=False)


class APTOS(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)


def show_images(images, labels, dataset_name):
    num_images = len(images)
    rows = int(num_images / 5) + 1

    fig, axes = plt.subplots(rows, 5, figsize=(15, rows * 3))

    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            ax.imshow(images[i].permute(1, 2, 0))  # permute to (H, W, C) for displaying RGB images
            ax.set_title(f"Label: {labels[i]}")
        ax.axis("off")

    plt.savefig(f'{dataset_name}_visualization.png')


def visualize_random_samples_from_clean_dataset(dataset, dataset_name):
    print(f"Start visualization of clean dataset: {dataset_name}")
    # Choose 20 random indices from the dataset
    if len(dataset) > 20:
        random_indices = random.sample(range(len(dataset)), 20)
    else:
        random_indices = [i for i in range(len(dataset))]

    random_samples = [dataset[i] for i in random_indices]

    images, labels = zip(*random_samples)

    print(f"len(labels): {len(labels)}")
    print(f"type(labels): {type(labels)}")
    print(f"type(labels[0]): {type(labels[0])}")
    print(f"labels[0]: {labels[0]}")
    labels = torch.tensor(labels)

    # Show the 20 random samples
    show_images(images, labels, dataset_name)
