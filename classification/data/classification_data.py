import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from dinov2.data.transforms import FUS13M_MEAN, FUS13M_STD

from classification.data.fetal_planes_dataset import FetalPlanesDataset
from classification.data.sono41_dataset import Sono41Dataset


# DataModule
class ClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        img_size=(224, 224),
        batch_size=32,
        num_workers=4,
        in_chans=1,
        normalization: str = "fus",
    ):
        super().__init__()

        self.num_workers = num_workers
        self.img_size = img_size
        self.batch_size = batch_size
        self.in_chans = in_chans

        if normalization == "fus":
            self.norm_mean = [FUS13M_MEAN] * in_chans
            self.norm_std = [FUS13M_STD] * in_chans
        elif normalization == "imagenet":
            self.norm_mean = [0.485, 0.456, 0.406]
            self.norm_std = [0.229, 0.224, 0.225]
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        self.train_transform, self.val_transform = self._get_transforms()
        self.dataset_class = self._parse_dataset_name(dataset_name)

    def _parse_dataset_name(self, dataset_name: str):
        if dataset_name == "Sono41":
            class_ = Sono41Dataset
        elif dataset_name == "FetalPlanes":
            class_ = FetalPlanesDataset
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"dataset_name: {dataset_name}")

        return class_  # ,kwargs

    def setup(self, stage=None):
        # Create instances of the datasets
        self.train_dataset = self.dataset_class(
            split="train", transform=self.train_transform, in_channels=self.in_chans
        )
        self.val_dataset = self.dataset_class(split="val", transform=self.val_transform, in_channels=self.in_chans)
        self.test_dataset = self.dataset_class(split="test", transform=self.val_transform, in_channels=self.in_chans)

        # Set the number of classes from the dataset
        self.num_classes = self.train_dataset.num_classes
        self.class_labels = self.train_dataset.class_labels

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _get_transforms(self):
        # Define the Albumentations transforms
        train_transform = A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), p=0.5),
                # A.Affine(
                #    scale=(0.8, 1.2),
                #    translate_percent=(0.2, 0.2),
                #    rotate=(-30, 30),
                #    shear=(-15, 15),
                #    interpolation=1,  # cv2.INTER_LINEAR
                #    mode=1,  # cv2.BORDER_REFLECT_101
                #    fit_output=True,
                #    p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
                # A.ToRGB(num_output_channels=self.in_chans),
                A.Normalize(mean=self.norm_mean, std=self.norm_std),
                ToTensorV2(),
            ]
        )

        val_transform = A.Compose(
            [
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
                # A.ToRGB(num_output_channels=self.in_chans),
                A.Normalize(mean=self.norm_mean, std=self.norm_std),
                ToTensorV2(),
            ]
        )

        return train_transform, val_transform
