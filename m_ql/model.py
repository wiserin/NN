import torch
import torch.nn as nn
import torch.optim as optim
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
# from .game import Game

class QModel(nn.Module):
    def __init__(self):
        super(QModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3200, 64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0,2),
            nn.Linear(64, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0,2),
            nn.Linear(128, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0,2),
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 25),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = QModel()
    summary(model, input_size=(1, 5, 5))