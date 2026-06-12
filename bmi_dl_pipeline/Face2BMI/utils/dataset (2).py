
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size=224, mode="train"):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if mode == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.03,
                               scale_limit=0.05,
                               rotate_limit=5, p=0.3),
            A.GaussNoise(p=0.15),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])


def get_tta_transforms(img_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return [
        A.Compose([A.Resize(img_size, img_size),
                   A.Normalize(mean=mean, std=std),
                   ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size),
                   A.HorizontalFlip(p=1.0),
                   A.Normalize(mean=mean, std=std),
                   ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size),
                   A.RandomBrightnessContrast(
                       brightness_limit=0.1,
                       contrast_limit=0.1, p=1.0),
                   A.Normalize(mean=mean, std=std),
                   ToTensorV2()]),
    ]


class BMIDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.loc[idx]
        img_path = f"{self.img_dir}/{row['id']}.png"
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bmi = torch.tensor(float(row['bmi']),
                           dtype=torch.float32)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, bmi


def get_sampler(dataframe):
    """Oversample extreme BMI + female samples."""
    bmi    = dataframe["bmi"].values
    bins   = [0, 18.5, 25, 30, 35, 100]
    labels = np.digitize(bmi, bins) - 1
    labels = np.clip(labels, 0, len(bins) - 2)

    # Base weights from BMI range
    class_counts   = np.bincount(labels, minlength=len(bins)-1)
    class_weights  = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels].copy()

    # Extra boost for female samples (6% of dataset)
    if "sex" in dataframe.columns:
        female_mask = (dataframe["sex"].str.lower() == "female").values
        sample_weights[female_mask] *= 3.0

    return WeightedRandomSampler(
        weights     = torch.tensor(sample_weights,
                                   dtype=torch.float32),
        num_samples = len(dataframe),
        replacement = True
    )
