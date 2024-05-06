import os

import torch
from glob import glob
import random
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms



def three_digits(a: int):
    x = str(a)
    if len(x) == 1:
        return f'00{a}'
    if len(x) == 2:
        return f'0{a}'
    return x


class WBCDataset(torch.utils.data.Dataset):
    def __init__(self, root1, root2,
                 labels1: pd.DataFrame, labels2: pd.DataFrame, transform=None, train=True, test_id=1, ratio=0.7):
        self.transform = transform
        self.root1 = root1
        self.root2 = root2
        self.labels1 = labels1
        self.labels2 = labels2
        self.train = train
        self.test_id = test_id
        self.targets = []
        labels1 = labels1[labels1['class label'] != 5]
        labels2 = labels2[labels2['class'] != 5]

        normal_df = labels1[labels1['class label'] == 1]
        self.normal_paths = [os.path.join(root1, f'{three_digits(x)}.bmp') for x in list(normal_df['image ID'])]
        random.seed(42)
        random.shuffle(self.normal_paths)
        self.separator = int(ratio * len(self.normal_paths))
        self.train_paths = self.normal_paths[:self.separator]

        if self.train:
            self.image_paths = self.train_paths
            self.targets = [0] * len(self.image_paths)
        else:
            if self.test_id == 1:
                all_images = glob(os.path.join(root1, '*.bmp'))
                self.image_paths = [x for x in all_images if x not in self.train_paths]
                self.image_paths = [x for x in self.image_paths if int(os.path.basename(x).split('.')[0]) in labels1['image ID'].values]
                ids = [os.path.basename(x).split('.')[0] for x in self.image_paths]
                ids_labels = list(labels1[labels1['image ID'] == int(x)]['class label'] for x in ids)
                self.targets = [0 if x.item() == 1 else 1 for x in ids_labels]
            else:
                self.image_paths = glob(os.path.join(root2, '*.bmp'))
                self.image_paths = [x for x in self.image_paths if int(os.path.basename(x).split('.')[0])
                                    in labels2['image ID'].values]
                self.targets = [
                    0 if labels2[labels2['image ID'] == int(os.path.basename(x).split('.')[0])]['class'].item() == 1
                    else 1 for x in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
