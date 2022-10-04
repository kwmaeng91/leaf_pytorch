import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_image
from PIL import Image

IMAGE_SIZE = 28

def calc_loss(out, y):
    return nn.CrossEntropyLoss()(out, y)

def calc_pred(out):
    _, pred = torch.max(out.data, 1)
    return pred

class ClientModel(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__()

        self.nn = nn.Sequential(
                ConvLayer(1, 32),
                ConvLayer(32, 64)
        )
        self.fc1 = nn.Linear(64 * 16, 2048)
        self.fc2 = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        x = self.nn(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 5)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data_x = [torch.Tensor(x).reshape(1, IMAGE_SIZE, IMAGE_SIZE) for x in data['x']]
        self.data_y = data['y']

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, i):
        return self.data_x[i], self.data_y[i]
