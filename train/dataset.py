import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional
import cv2


class invoiceD(Dataset):
    """
    Image dataset for invoice data
    """
    def __init__(self, input_folder, resize):
        super().__init__()
        self.input_folder = input_folder
        self.resize = resize
        self.data, self.labels = self.load_data()

        
    def __getitem__(self, idx):
        image = self.data[idx]
        labels = self.data[idx]
        return image, labels

    def __len__(self):
        return len(self.data)

    def load_data(self):
        data = []
        labels = []

    
        data = np.array(data)
        labels = np.array(labels)

        return data, labels