from os.path import join
from datasets.cifar100 import CIFAR100
import torchvision.transforms as transforms

DATASETS = {
    'c100': CIFAR100,
}

def transform_set(train, img_size, crop_pad, flip):
    transform = []
    transform.append(transforms.Resize((img_size, img_size)))
    transform.append(transforms.Pad(crop_pad))

    if train:
        transform.append(transforms.RandomCrop((img_size, img_size)))
        if flip: transform.append(transforms.RandomHorizontalFlip(p=0.5))
    else:
        transform.append(transforms.CenterCrop((img_size, img_size)))

    transform = transforms.Compose(transform)
    return transform

def build_data(data_name, data_path, train, trigger, transform):
    data = DATASETS[data_name](root=join(data_path, DATASETS[data_name].__name__.lower()), train=train, trigger=trigger, transform=transform)
    return data
