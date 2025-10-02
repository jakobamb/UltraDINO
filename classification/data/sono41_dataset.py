import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class Sono41Dataset(Dataset):
    def __init__(self, split, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")
        elif split == "train":
            csv_file = "sono41/train.csv"
        elif split == "val":
            csv_file = "sono41/val.csv"
        elif split == "test":
            csv_file = "sono41/test.csv"

        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

        # Ensure labels are integers
        self.dataframe["label_sono_41"] = self.dataframe["label_sono_41"].astype(int)

        # Get the unique labels and determine the number of classes
        self.classes = sorted(self.dataframe["label_sono_41"].unique())
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the image path and label from the dataframe
        img_path = self.dataframe.iloc[idx]["path"]
        label = self.dataframe.iloc[idx]["label_sono_41"]

        # Load the image
        image = np.array(Image.open(img_path).convert("L"))

        # Apply transformations if any
        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
