from train.dataset import invoiceD
import torch

data = invoiceD(r'data\Invoice_data.v12i.multiclass', 256)
data.data