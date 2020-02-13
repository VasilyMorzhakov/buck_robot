import torch
import torch.nn as nn
import torch.nn.functional as f
import config

class Detector(nn.Module):
    def __init__(self,n_class):
        super().__init__()
        self.n_class=n_class
        self.conv1=nn.Conv2d(3, 32, 5, padding=2)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 1, padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5=nn.Conv2d(256,n_class,1,padding=0)
        return
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=f.relu(x)
        x=nn.MaxPool2d(2)(x)
        x=self.conv2(x)
        x = self.bn2(x)
        x = f.relu(x)
        x=nn.MaxPool2d(2)(x)
        x=self.conv3(x)
        x = self.bn3(x)
        x=f.relu(x)
        x=nn.MaxPool2d(2)(x)
        x=self.conv4(x)
        x = self.bn4(x)
        x=f.relu(x)
        x=self.conv5(x)
        return x



