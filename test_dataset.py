from train.dataset import InvoiceDataset
from train.train import train_model
import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim
dataset = InvoiceDataset(r'C:\Users\ombeh\Downloads\crabsw\data\Invoice_data.v12i.multiclass', resize=(224, 225), split='train')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cri = CrossEntropyLoss()
model = train_model(dataset, cri, device)