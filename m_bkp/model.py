import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from db.requests import load_dataset_X



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Первый сверточный слой с 1 каналом на входе и 32 на выходе. Размер ядра свертки 3х3
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 2, 3, 1), 
            nn.ReLU(inplace=True)
        )

        # 2 полносвязных слоя NN и softmax
        self.fc = nn.Sequential(
            nn.Flatten(), # Выравниваем в плоский вектор
            nn.Linear(18, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 25)
        )

    # Прямой проход
    def forward(self, x):
        x = self.layer1(x)
        x = self.fc(x)
        return x


def NN_bkp():
    # Загрузка обучающего и тестового наборов
    train_dataset, test_dataset = load_dataset_X()

    # Загрузка данных в DataLoader для мини-батчей
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Инициализация модели
    model = CNNModel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    epochs = 50
    for epoch in range(epochs):
        model.train()  # перевод в режим обучения
        running_loss = 0.0
        
        for fields, labels in train_loader:
            fields, labels = fields.to(device), labels.to(device)  # перенос данных на GPU

            optimizer.zero_grad()  # обнуление градиентов
            outputs = model(fields)  # прямой проход
            loss = criterion(outputs, labels)  # вычисление ошибки
            loss.backward()  # обратный проход
            optimizer.step()  # шаг оптимизации

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")


    model.eval()  # перевод модели в режим оценки
    correct = 0
    total = 0

    with torch.no_grad():  # отключение градиентов для оценки
        for fields, labels in test_loader:
            fields, labels = fields.to(device), labels.to(device)
            outputs = model(fields)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

