import os
import numpy as np
import pickle
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class CIFAR100(data.Dataset):
    base_folder = 'cifar-100-python'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(self, root, train=True, trigger=None, transform=None):
        super(CIFAR100, self).__init__()
        self.root = root
        self.trigger = trigger
        self.transform = transform
        file_list = self.train_list if train else self.test_list
        self.data, self.targets = [], []
        for file_name, checksum in file_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        backdoor, source = 0, target
        img = Image.fromarray(img)
        if self.trigger is not None: img, target, backdoor = self.trigger(img, target, backdoor, idx)
        if self.transform is not None: img = self.transform(img)
        img = self.toTensor(img)
        return img, target, backdoor, source, idx

    def __len__(self):
        return self.data.shape[0]
