import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# Fetal planes dataset from Burgos-Artizzu et al.
# https://zenodo.org/records/3904280
class FetalPlanesDataset(Dataset):
    def __init__(self, split, transform=None, in_channels=1):
        self.split = split
        self.transform = transform
        self.in_channels = in_channels
        self.root = os.path.join(os.getenv("DOWNSTREAM_DATA"), "fetal_planes_db")

        self.num_classes = 6
        self.class_labels = [
            "Other",
            "Maternal cervix",
            "Fetal abdomen",
            "Fetal brain",
            "Fetal femur",
            "Fetal thorax",
        ]

        # Load the dataset
        self.paths, self.labels = self._load_paths_labels(self.root)

    def get_image_data(self, index: int) -> Image:
        path = self.paths[index]

        img = Image.open(path)
        if img.mode != "L":
            img = img.convert("L")

        if self.in_channels > 1:
            img = np.array(img)
            img = np.stack([img] * self.in_channels, axis=-1)

        return img

    def get_target(self, index: int) -> int:
        return self.labels[index]

    def _load_paths_labels(self, root) -> Tuple[List[str], List[int]]:
        csv_file = os.path.join(os.getenv("PROJECT_DIR"), f"data/fetal_planes_db/{self.split}_split.csv")
        df = pd.read_csv(csv_file, sep=",")
        df = df.rename(columns=lambda x: x.strip())

        df["label"] = df["Plane"].apply(lambda x: self.class_labels.index(x))
        assert len(df["label"].unique()) == self.num_classes

        df["path"] = df["Image_name"].apply(lambda x: os.path.join(root, "Images", f"{x}.png"))

        return list(df["path"]), list(df["label"])

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"Cannot read image for sample {index}") from e
        image = np.array(image)

        target = self.get_target(index)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, target

    def __len__(self) -> int:
        return len(self.paths)
