import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml

from server.models.emotionModel import EmotionModel


def readConfig(path: str):
    path_to_config = resource_path(path)
    configs = load_config(path_to_config)
    return configs

def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def resource_path(relative_path) -> str:
    """Возвращает корректный путь для доступа к ресурсам после сборки .exe"""
    #if getattr(sys, 'frozen', False):
    try:
        # PyInstaller создаёт временную папку _MEIPASS для ресурсов
        base_path = sys._MEIPASS
    except Exception:
        # Если приложение запущено из исходного кода, то используется обычный путь
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def main():
    configs = readConfig(os.path.join('configs', 'emotion.yaml'))

    # Предобработка данных
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    # Загрузчики данных
    train_dataset = datasets.ImageFolder(os.path.join(configs['dataset_root'], 'train'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    val_dataset = datasets.ImageFolder(os.path.join(configs['dataset_root'], 'test'), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Модель, функция потерь, оптимизатор
    model = EmotionModel()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)

    train_model(model, train_loader, criterion, optimizer, num_epochs=1)
    rel_path = configs['emotion_model_path']
    abs_path = resource_path(rel_path)

    torch.save(model.state_dict(), os.path.join(abs_path, 'emotion_model.pth') )


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

if __name__ == '__main__':
    main()