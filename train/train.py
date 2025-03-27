import torch
import numpy as np
from hestreg.train.models.resnet import resnet
from torch.utils.data import DataLoader

def train_model(model, dataset, criterian, optimizer, device):
    batch = 64
    num_epoch = 25

    labels = dataset.labels

    train_dataset = [(torch.tensor(dataset.data[idx], dtype=torch.float32), torch.tensor(np.long(labels[idx][0]), dtype=torch.long)) for idx in range(len(dataset))]
    print(len(dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    model = resnet(model, train_dataloader)

    model.to(device)

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