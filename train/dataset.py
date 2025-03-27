import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional
import cv2


class invoiceD(Dataset):
    def __init__(self, input_folder, resize, split='train'):
        super().__init__()
        self.input_folder = input_folder
        self.resize = resize
        self.split = split
        self.data, self.labels = self.load_data()


    def __getitem__(self, idx):
        image = self.train[idx]
        label = self.labels[idx]

        # Apply resize if specified
        if self.resize:
            image = cv2.resize(image, self.resize)

        # Convert to tensor if needed
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

        # Path to the data folder based on the split (train, test, valid)
        data_path = os.path.join(self.input_folder, self.split)

        if not os.path.exists(data_path):
            raise ValueError(f"Data path does not exist: {data_path}")

        # Get all files in the directory
        for label_folder in os.listdir(data_path):
            label_path = os.path.join(data_path, label_folder)

            # Skip if not a directory
            if not os.path.isdir(label_path):
                continue

            # Process each image in the label folder
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)

                # Skip if not a file
                if not os.path.isfile(img_path):
                    continue

                # Load the image
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Warning: Could not load image {img_path}")
                        continue

                    # Convert BGR to RGB (OpenCV loads as BGR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Add to dataset
                    data.append(image)
                    labels.append(label_folder)  # Using folder name as label
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

        data = np.array(data)
        labels = np.array(labels)

        print(f"Loaded {len(data)} images for {self.split} split")

        return data, labels