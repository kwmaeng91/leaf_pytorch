import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_image
from PIL import Image

IMAGE_SIZE = 84
IMAGES_DIR = os.path.join('/media/sf_vbox_shared/data/', 'celeba', 'data', 'raw', 'img_align_celeba')

def calc_loss(out, y):
    return nn.CrossEntropyLoss()(out, y)

def calc_pred(out):
    _, pred = torch.max(out.data, 1)
    return pred

class ClientModel(nn.Module):
    def __init__(self, num_classes):
        super(ClientModel, self).__init__()
        self.nn = nn.Sequential(
                ConvLayer(3, 32),
                ConvLayer(32, 32),
                ConvLayer(32, 32),
                ConvLayer(32, 32)
        )
        self.fc = nn.Linear(32*9, num_classes)

    def forward(self, x):
        x = self.nn(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3)
        self.bn = nn.BatchNorm2d(out_channel) # TODO: Do we want bn?
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.pool(self.bn(self.conv(x))))


class CustomDataset(Dataset):
    def __init__(self, data):
        data_x = [self._load_image(i) for i in data['x']]
        data_x = torch.stack(data_x)
        self.data_x = data_x
        self.data_y = data['y']

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, i):
        return self.data_x[i], self.data_y[i]

    def _load_image(self, img_name):
        image2tensor = torchvision.transforms.ToTensor()
        img = Image.open(os.path.join(IMAGES_DIR, img_name))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
        img = image2tensor(img) * 255
        return img

