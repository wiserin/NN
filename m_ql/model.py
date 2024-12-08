import torch
import torch.nn as nn
import torch.optim as optim
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .game import Game

class QModel(nn.Module):
    def __init__(self):
        super(QModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Flatten(), # Выравниваем в плоский вектор
            nn.Linear(800, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 25)

        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.fc(x)
        return x