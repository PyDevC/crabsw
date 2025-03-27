import torch
import numpy as np
from .model import OCRModel
from torch.utils.data import DataLoader
import torch.optim as optim

def train_model(dataset, criterian, device):
    batch = 64
    num_epoch = 25
    train_dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    model = OCRModel()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for input, label in train_dataloader:
            input = input.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterian(output, label)
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = correct / total * 100
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    return model