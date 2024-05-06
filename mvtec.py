import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from utils import transform_color, transform_resnet18, Transform


class IMAGENET30_TEST_DATASET(Dataset):
    def __init__(self, root_dir="/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/", transform=None):
        """
        Args:
            root_dir (string): Directory with all the classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_path_list = []
        self.targets = []

        # Map each class to an index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        # print(f"self.class_to_idx in ImageNet30_Test_Dataset:\n{self.class_to_idx}")

        # Walk through the directory and collect information about the images and their labels
        for i, class_name in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            for instance_folder in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance_folder)
                if instance_path != "/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/airliner/._1.JPEG":
                    for img_name in os.listdir(instance_path):
                        if img_name.endswith('.JPEG'):
                            img_path = os.path.join(instance_path, img_name)
                            # image = Image.open(img_path).convert('RGB')
                            self.img_path_list.append(img_path)
                            self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        image = default_loader(img_path)
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# This is a modified version of original  https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
# This file and the mvtec data directory must be in the same directory, such that:
# /.../this_directory/mvtecDataset.py
# /.../this_directory/mvtec/bottle/...
# /.../this_directory/mvtec/cable/...
# and so on

import os
import os.path
import numpy as np
import torch.utils.data as data
import matplotlib.image as mpimg
from torchvision import transforms
import random

from PIL import Image


def center_paste(large_img, small_img):
    # Calculate the center position
    # to_pil = transforms.ToPILImage()
    # small_img = to_pil(small_img)
    large_width, large_height = large_img.size
    small_width, small_height = small_img.size

    # Calculate the top-left position
    left = (large_width - small_width) // 2
    top = (large_height - small_height) // 2

    # Create a copy of the large image to keep the original unchanged
    result_img = large_img.copy()

    # Paste the small image onto the large one at the calculated position
    result_img.paste(small_img, (left, top))

    return result_img


from glob import glob


class MVTEC(data.Dataset):

    def __init__(self, root, train=True,
                 transform=None, transform_norm=None,
                 category='carpet', resize=None, use_imagenet=False,
                 select_random_image_from_imagenet=False, shrink_factor=0.9):
        self.root = root
        self.transform = transform
        self.image_files = []
        self.transform_norm = transform_norm
        print("category MVTecDataset:", category)
        if train:
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            self.image_files = image_files
        self.image_files.sort(key=lambda y: y.lower())
        self.train = train
        self.resize = resize
        self.use_imagenet = use_imagenet
        if use_imagenet:
            self.resize = int(resize * shrink_factor)
        self.select_random_image_from_imagenet = select_random_image_from_imagenet
        self.imagenet30_testset = IMAGENET30_TEST_DATASET()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        to_pil = transforms.ToPILImage()
        if not self.train:
            image = to_pil(image)

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1

        if self.train:
            return image, target

        if self.select_random_image_from_imagenet:
            imagenet30_img = self.imagenet30_testset[int(random.random() * len(self.imagenet30_testset))][0].resize(
                (224, 224))
        else:
            imagenet30_img = self.imagenet30_testset[100][0].resize((224, 224))

        # if resizing image
        if self.use_imagenet and self.resize is not None:
            resizeTransf = transforms.Resize((self.resize, self.resize))
            image = resizeTransf(image)

        #         print(f"imagenet30_img.size: {imagenet30_img.size}")
        #         print(f"img.size: {img.size}")
        image = center_paste(imagenet30_img, image)

        to_trans = transforms.ToTensor()
        image = to_trans(image)
        image = self.transform_norm(image)

        return image, target

    def __len__(self):
        return len(self.image_files)


def get_loader_mvtec(batch_size, backbone, category):
    transform = transform_color if backbone == 152 else transform_resnet18
    transform_color2 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_data = MVTEC(root='/kaggle/input/mvtec-ad', train=True, transform=transform, category=category,
                       resize=224, use_imagenet=True, select_random_image_from_imagenet=True,
                       shrink_factor=1, transform_norm=transform_color2)
    test_data1 = MVTEC(root='/kaggle/input/mvtec-ad', train=False, transform=transform, category=category,
                       resize=224, use_imagenet=True, select_random_image_from_imagenet=True,
                       shrink_factor=1, transform_norm=transform_color2)
    test_data2 = MVTEC(root='/kaggle/input/mvtec-ad', train=False, transform=transform, category=category,
                       resize=224, use_imagenet=True, select_random_image_from_imagenet=True,
                       shrink_factor=0.9, transform_norm=transform_color2)

    train_data1 = MVTEC(root='/kaggle/input/mvtec-ad', train=True, transform=Transform(), category=category,
                        resize=224, use_imagenet=True, select_random_image_from_imagenet=True,
                        shrink_factor=1, transform_norm=transform_color2)

    print('len(train_data)', len(train_data))
    print('len(test_data1)', len(test_data1))
    print('len(test_data2)', len(test_data2))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2,
                                               drop_last=False)
    train_loader1 = torch.utils.data.DataLoader(train_data1, batch_size=batch_size, shuffle=True, num_workers=2,
                                                drop_last=False)
    test_loader1 = torch.utils.data.DataLoader(test_data1, batch_size=batch_size, shuffle=False, num_workers=2,
                                               drop_last=False)
    test_loader2 = torch.utils.data.DataLoader(test_data2, batch_size=batch_size, shuffle=False, num_workers=2,
                                               drop_last=False)

    return train_loader, test_loader1, train_loader1, test_loader2
