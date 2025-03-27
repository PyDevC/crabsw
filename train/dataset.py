import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional
import cv2


class InvoiceDataset(Dataset):
    def __init__(self, input_folder, resize=None, split='train'):
        super().__init__()
        self.input_folder = input_folder
        self.resize = resize
        self.split = split
        self.data, self.labels = self.load_data()  # Load images from 'train' folder

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.resize:
            image = cv2.resize(image, self.resize)

        image = functional.to_tensor(image)
        return image, label

    def __len__(self):
        return len(self.data)

    def load_data(self):
        """
        Load image data from the specified input folder
        """
        data = []
        labels = []

        data_path = os.path.join(self.input_folder, self.split)
        print(f"Looking for images in: {data_path}")

        if not os.path.exists(data_path):
            raise ValueError(f"Data path does not exist: {data_path}")

        # Process images directly instead of expecting label folders
        for img_file in os.listdir(data_path):
            img_path = os.path.join(data_path, img_file)

            if not os.path.isfile(img_path):
                print(f"Skipping {img_file}, not a file.")
                continue  # ✅ Move continue inside the condition

            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue  # ✅ Skip corrupted images

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                data.append(image)
                labels.append("unknown")  # Use a placeholder label if none exists
                print(f"Loaded: {img_path}")

            except Exception as e:
                print(f"Error loading {img_path}: {e}")

        print(f"Loaded {len(data)} images for '{self.split}' split")
        return data, labels  # ✅ Correct indentation (inside method)