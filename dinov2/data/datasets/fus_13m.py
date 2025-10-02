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


logger = logging.getLogger("dinov2")


class FUS13M(ExtendedVisionDataset):
    def __init__(
        self,
        split: Literal["train", "test"] = "train",
        # root: str,
        # extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(os.getenv("PRETRAIN_DATA"), transforms, transform, target_transform)
        self.split = split

        if self.split.lower() == "train":
            pathsfile = "train_paths_13m.csv"
        elif self.split == "test":
            pathsfile = "test_paths_13m.csv"

        self.paths = self._load_paths(os.path.join(self.root, pathsfile))

    def get_image_data(self, index: int) -> bytes:
        path = os.path.join(self.root, self.paths[index])

        img = Image.open(path)
        if img.mode != "L":
            img = img.convert("L")

        return img

    def get_target(self, index: int) -> Any:
        return None  # SSL dataset

    def _load_paths(self, paths_file) -> List[str]:
        with open(paths_file, "r") as f:
            return [line.strip() for line in f]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.paths)
