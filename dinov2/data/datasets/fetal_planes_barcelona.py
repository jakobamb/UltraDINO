# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import numpy as np

from PIL import ImageFilter, Image

from .extended import ExtendedVisionDataset

import pandas as pd


logger = logging.getLogger("dinov2")


# TODO: Implement validation dataset. K fold split?
class FetalPlanesBarcelona(ExtendedVisionDataset):
    def __init__(
        self,
        root: str = "/data/proto/Jakob/public_datasets/FETAL_PLANES_DB/",
        split: Literal["train", "test"] = "train",
        # extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(None, transforms, transform, target_transform)
        self.split = split

        self.root = root
        paths, labels = self._load_paths_labels(root)

        self.paths = paths
        self.labels = labels

    def get_image_data(self, index: int) -> bytes:
        path = self.paths[index]

        img = Image.open(path)
        if img.mode != "L":
            img = img.convert("L")

        return img

    def get_target(self, index: int) -> Any:
        return self.labels[index]

    def _load_paths_labels(self, root):
        csv_file = os.path.join(root, "FETAL_PLANES_DB_data.csv")
        df = pd.read_csv(csv_file, sep=";")
        df.rename(columns=lambda x: x.strip())

        # label mapping for reproducability
        labels_dict = {
            "Other": 0,
            "Maternal cervix": 1,
            "Fetal abdomen": 2,
            "Fetal brain": 3,
            "Fetal femur": 4,
            "Fetal thorax": 5,
        }

        df["label"] = df["Plane"].apply(lambda x: labels_dict[x])  # remove trailing whitespaces
        assert len(df["label"].unique()) == 6

        df["path"] = df["Image_name"].apply(lambda x: os.path.join(root, "Images", x))

        if self.split == "train":
            df = df[df["Train"] == 1]
        elif self.split == "test":
            df = df[df["Train"] == 0]
        else:
            raise ValueError(f"invalid split: {self.split}")

        return list(df["path"]), list(df["label"])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"Cannot read image for sample {index}") from e
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.paths)
