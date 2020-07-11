import numpy as np
import random
from PIL import Image
import torch
import matplotlib.pyplot as plt

import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import time


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class Create_data(Dataset):
    def __init__(self, dataset, augment=False):
        super(Create_data, self).__init__()
        self.dataset = dataset
        self.num_train = len(dataset.imgs)
        self.augment = augment
        self.image_size = (224, 224)

    def __len__(self):
        return self.num_train

    def __getitem__(self, index):
        # image_label = random.choice(self.dataset.imgs)
        image_label = self.dataset.imgs[index]
        image = Image.open(image_label[0])
        image = image.resize(size=self.image_size, resample=Image.BICUBIC)

        if self.augment:
            flip = rand() < .5
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image = np.array(image, dtype=np.float32) / 255.
        image1 = transforms.ToTensor()(image)
        y = torch.from_numpy(np.array([image_label[1]], dtype=np.float32))
        return image1, y


def get_train_valid_loader(full_dataset, batch_size, ratio=0.8, num_workers=4, pin_memory=True):
    train_size = int(ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
    )
    valid_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
    )
    return train_loader, valid_loader


def get_test_loader(test_dataset, batch_size, num_workers=4):
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
    )
    return test_loader


if __name__ == "__main__":
    batch_size = 128
    ratio = 0.8
    num_workers = 4
    pin_memory = True
    train_dataset = dset.ImageFolder(root='/home/jackzhou/PycharmProjects/CBRSIR_hash/Dataset/train')
    full_dataset = Create_data(train_dataset, augment=False)
    train_loader, valid_loader = get_train_valid_loader(full_dataset, batch_size, ratio, num_workers, pin_memory)
    time_start = time.time()
    for i in range(2):
        for i, (x, y) in enumerate(train_loader):
            print(x.shape, y.shape)
    time_end = time.time()
    print('totally cost', time_end - time_start)

    print(len(train_dataset.class_to_idx))

