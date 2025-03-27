from train.dataset import invoiceD
from train.train import train_model
import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim


data = invoiceD(r'data\Invoice_data.v12i.multiclass', 256)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cri = CrossEntropyLoss()
model = train_model(cri,)