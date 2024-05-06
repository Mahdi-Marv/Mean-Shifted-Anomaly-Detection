import os
import shutil
from pathlib import Path

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

from wbc import WBCDataset

BICUBIC = InterpolationMode.BICUBIC

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from glob import glob
import pandas as pd
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


class BrainTest(torch.utils.data.Dataset):
    def __init__(self, transform, test_id=1):

        self.transform = transform
        self.test_id = test_id

        test_normal_path = glob('./Br35H/dataset/test/normal/*')
        test_anomaly_path = glob('./Br35H/dataset/test/anomaly/*')

        self.test_path = test_normal_path + test_anomaly_path
        self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

        if self.test_id == 2:
            test_normal_path = glob('./brats/dataset/test/normal/*')
            test_anomaly_path = glob('./brats/dataset/test/anomaly/*')

            self.test_path = test_normal_path + test_anomaly_path
            self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

    def __len__(self):
        return len(self.test_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.test_path[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        has_anomaly = 0 if self.test_label[idx] == 0 else 1

        # return img, , has_anomaly, img_path
        return img, has_anomaly


class BrainTrain(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.image_paths = glob('./Br35H/dataset/train/normal/*')
        brats_mod = glob('./brats/dataset/train/normal/*')
        random.seed(1)
        random_brats_images = random.sample(brats_mod, 50)
        self.image_paths.extend(random_brats_images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, 0


def prepare_br35h_dataset_files():
    normal_path35 = '/kaggle/input/brain-tumor-detection/no'
    anomaly_path35 = '/kaggle/input/brain-tumor-detection/yes'

    print(f"len(os.listdir(normal_path35)): {len(os.listdir(normal_path35))}")
    print(f"len(os.listdir(anomaly_path35)): {len(os.listdir(anomaly_path35))}")

    Path('./Br35H/dataset/test/anomaly').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('./Br35H/dataset/test/anomaly')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/test/anomaly', f))

    anom35 = os.listdir(anomaly_path35)
    for f in anom35:
        shutil.copy2(os.path.join(anomaly_path35, f), './Br35H/dataset/test/anomaly')


    normal35 = os.listdir(normal_path35)
    random.shuffle(normal35)
    ratio = 0.7
    sep = round(len(normal35) * ratio)

    Path('./Br35H/dataset/test/normal').mkdir(parents=True, exist_ok=True)
    Path('./Br35H/dataset/train/normal').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('./Br35H/dataset/test/normal')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/test/normal', f))

    flist = [f for f in os.listdir('./Br35H/dataset/train/normal')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/train/normal', f))

    for f in normal35[:sep]:
        shutil.copy2(os.path.join(normal_path35, f), './Br35H/dataset/train/normal')
    for f in normal35[sep:]:
        shutil.copy2(os.path.join(normal_path35, f), './Br35H/dataset/test/normal')


def prepare_brats2015_dataset_files():
    labels = pd.read_csv('/kaggle/input/brain-tumor/Brain Tumor.csv')
    labels = labels[['Image', 'Class']]
    labels.tail() # 0: no tumor, 1: tumor

    labels.head()

    brats_path = '/kaggle/input/brain-tumor/Brain Tumor/Brain Tumor'
    lbl = dict(zip(labels.Image, labels.Class))

    keys = lbl.keys()
    normalbrats = [x for x in keys if lbl[x] == 0]
    anomalybrats = [x for x in keys if lbl[x] == 1]

    Path('./brats/dataset/test/anomaly').mkdir(parents=True, exist_ok=True)
    Path('./brats/dataset/test/normal').mkdir(parents=True, exist_ok=True)
    Path('./brats/dataset/train/normal').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('./brats/dataset/test/anomaly')]
    for f in flist:
        os.remove(os.path.join('./brats/dataset/test/anomaly', f))

    flist = [f for f in os.listdir('./brats/dataset/test/normal')]
    for f in flist:
        os.remove(os.path.join('./brats/dataset/test/normal', f))

    flist = [f for f in os.listdir('./brats/dataset/train/normal')]
    for f in flist:
        os.remove(os.path.join('./brats/dataset/train/normal', f))

    ratio = 0.7
    random.shuffle(normalbrats)
    bratsep = round(len(normalbrats) * ratio)

    for f in anomalybrats:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), './brats/dataset/test/anomaly')
    for f in normalbrats[:bratsep]:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), './brats/dataset/train/normal')
    for f in normalbrats[bratsep:]:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), './brats/dataset/test/normal')


def get_loader_wbc(batch_size, backbone):
    transform = transform_color if backbone == 152 else transform_resnet18

    df1 = pd.read_csv('/kaggle/working/segmentation_WBC/Class Labels of Dataset 1.csv')
    df2 = pd.read_csv('/kaggle/working/segmentation_WBC/Class Labels of Dataset 2.csv')

    test_set1 = WBCDataset('/kaggle/working/segmentation_WBC/Dataset 1', '/kaggle/working/segmentation_WBC/Dataset 2',
                           df1, df2, transform=transform, train=False, test_id=1)
    test_set2 = WBCDataset('/kaggle/working/segmentation_WBC/Dataset 1', '/kaggle/working/segmentation_WBC/Dataset 2',
                           df1, df2, transform=transform, train=False, test_id=2)
    train_set = WBCDataset('/kaggle/working/segmentation_WBC/Dataset 1', '/kaggle/working/segmentation_WBC/Dataset 2',
                           df1, df2, transform=transform, train=True)
    train_set1 = WBCDataset('/kaggle/working/segmentation_WBC/Dataset 1', '/kaggle/working/segmentation_WBC/Dataset 2',
                           df1, df2, transform=Transform(), train=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                                               drop_last=False)
    train_loader1 = torch.utils.data.DataLoader(train_set1, batch_size=batch_size, shuffle=True, num_workers=2,
                                                drop_last=False)
    test_loader_main = torch.utils.data.DataLoader(test_set1, batch_size=batch_size, shuffle=False, num_workers=2,
                                                   drop_last=False)
    test_loader_shift = torch.utils.data.DataLoader(test_set2, batch_size=batch_size, shuffle=False, num_workers=2,
                                                    drop_last=False)

    return train_loader, test_loader_main, train_loader1, test_loader_shift



def get_loader_brain(batch_size, backbone):
    prepare_br35h_dataset_files()
    prepare_brats2015_dataset_files()
    transform = transform_color if backbone == 152 else transform_resnet18


    train_data = BrainTrain(transform=transform)
    train_data1 = BrainTrain(transform=Transform())

    test_data1 = BrainTest(transform=transform, test_id=1)
    test_data2 = BrainTest(transform=transform, test_id=2)

    visualize_random_samples_from_clean_dataset(train_data, "trainset")
    visualize_random_samples_from_clean_dataset(test_data1, "testset2")
    visualize_random_samples_from_clean_dataset(test_data2, "testset2")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2,
                                               drop_last=False)
    train_loader1 = torch.utils.data.DataLoader(train_data1, batch_size=batch_size, shuffle=True, num_workers=2,
                                                drop_last=False)
    test_loader_main = torch.utils.data.DataLoader(test_data1, batch_size=batch_size, shuffle=False, num_workers=2,
                                                   drop_last=False)
    test_loader_shift = torch.utils.data.DataLoader(test_data2, batch_size=batch_size, shuffle=False, num_workers=2,
                                                    drop_last=False)
    return train_loader, test_loader_main, train_loader1, test_loader_shift



class Waterbird(torch.utils.data.Dataset):
    def __init__(self, root, df, transform, train=True, count_train_landbg=-1, count_train_waterbg=-1, mode='bg_all',
                 count=-1):
        self.transform = transform
        self.train = train
        self.df = df
        lb_on_l = df[(df['y'] == 0) & (df['place'] == 0)]
        lb_on_w = df[(df['y'] == 0) & (df['place'] == 1)]
        self.normal_paths = []
        self.labels = []

        normal_df = lb_on_l.iloc[:count_train_landbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_landbg])
        normal_df = lb_on_w.iloc[:count_train_waterbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_waterbg])

        if train:
            self.image_paths = self.normal_paths
        else:
            self.image_paths = []
            if mode == 'bg_all':
                dff = df
            elif mode == 'bg_water':
                dff = df[(df['place'] == 1)]
            elif mode == 'bg_land':
                dff = df[(df['place'] == 0)]
            else:
                print('Wrong mode!')
                raise ValueError('Wrong bg mode!')
            all_paths = dff[['img_filename', 'y']].to_numpy()
            for i in range(len(all_paths)):
                full_path = os.path.join(root, all_paths[i][0])
                if full_path not in self.normal_paths:
                    self.image_paths.append(full_path)
                    self.labels.append(all_paths[i][1])


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if self.train:
            return img, 0
        else:
            return img, self.labels[idx]


def get_loader_aptos(batch_size, backbone):
    train_normal_path = glob('/kaggle/working/Mean-Shifted-Anomaly-Detection/APTOS/train/NORMAL/*')
    train_normal_label = [0] * len(train_normal_path)

    test_normal_path = glob('/kaggle/working/Mean-Shifted-Anomaly-Detection/APTOS/test/NORMAL/*')
    test_anomaly_path = glob('/kaggle/working/Mean-Shifted-Anomaly-Detection/APTOS/test/ABNORMAL/*')

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
    test_loader_2 = second_dataset(transform, batch_size)

    return train_loader, test_loader_2, torch.utils.data.DataLoader(train_set1, batch_size=batch_size,
                                                                  shuffle=True, num_workers=2, drop_last=False)


class ISIC2018(Dataset):
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


def get_loader_waterbirds(batch_size):
    import pandas as pd
    df = pd.read_csv('/kaggle/input/waterbird/waterbird/metadata.csv')
    root = '/kaggle/input/waterbird/waterbird'
    trainset = Waterbird(root=root, df=df, transform=transform_color, train=True,
                         count_train_landbg=3500, count_train_waterbg=100, mode='bg_all')

    trainset1 = Waterbird(root=root, df=df, transform=Transform(), train=True,
                         count_train_landbg=3500, count_train_waterbg=100, mode='bg_all')

    testset_land = Waterbird(root=root, df=df, transform=transform_color, train=False,
                            count_train_landbg=3500, count_train_waterbg=100, mode='bg_land')

    testset_water = Waterbird(root=root, df=df, transform=transform_color, train=False,
                             count_train_landbg=3500, count_train_waterbg=100, mode='bg_water')

    visualize_random_samples_from_clean_dataset(trainset, "trainset")
    # visualize_random_samples_from_clean_dataset(trainset1, "trainset1")
    visualize_random_samples_from_clean_dataset(testset_land, "testset_land")
    visualize_random_samples_from_clean_dataset(testset_water, "testset_water")

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                  drop_last=False)
    train_loader1 = torch.utils.data.DataLoader(trainset1, batch_size=batch_size, shuffle=True, num_workers=2,
                                                    drop_last=False)
    test_loader_land = torch.utils.data.DataLoader(testset_land, batch_size=batch_size, shuffle=False, num_workers=2,
                                                    drop_last=False)
    test_loader_water = torch.utils.data.DataLoader(testset_water, batch_size=batch_size, shuffle=False, num_workers=2,
                                                    drop_last=False)
    return train_loader, test_loader_land, train_loader1, test_loader_water



def second_dataset(transform, batch_size):
    df = pd.read_csv('/kaggle/input/ddrdataset/DR_grading.csv')
    label = df["diagnosis"].to_numpy()
    path = df["id_code"].to_numpy()

    normal_path = path[label == 0]
    anomaly_path = path[label != 0]

    shifted_test_path = list(normal_path) + list(anomaly_path)
    shifted_test_label = [0] * len(normal_path) + [1] * len(anomaly_path)

    shifted_test_path = ["/kaggle/input/ddrdataset/DR_grading/DR_grading/" + s for s in shifted_test_path]
    shifted_test_set = APTOS(image_path=shifted_test_path, labels=shifted_test_label, transform=transform)
    visualize_random_samples_from_clean_dataset(shifted_test_set, "shift testset")
    shifted_test_loader = torch.utils.data.DataLoader(shifted_test_set, shuffle=False, batch_size=batch_size)
    return shifted_test_loader




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

    # print(f"len(labels): {len(labels)}")
    # print(f"type(labels): {type(labels)}")
    # print(f"type(labels[0]): {type(labels[0])}")
    # print(f"labels[0]: {labels[0]}")
    labels = torch.tensor(labels)

    # Show the 20 random samples
    show_images(images, labels, dataset_name)
