from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from xray_preprocess import read_kaggle_image_as_rgb


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class KaggleChestXrayDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split,
        image_size=224,
        augment=False,
        harmonize_preprocess=True,
        crop_body=True,
        percentile_lower=1.0,
        percentile_upper=99.0,
        crop_threshold=5,
        crop_margin=10,
    ):
        self.csv_path = Path(csv_path)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        self.harmonize_preprocess = harmonize_preprocess
        self.crop_body = crop_body
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.crop_threshold = crop_threshold
        self.crop_margin = crop_margin

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Missing Kaggle split CSV: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        required_columns = {"path", "label", "split"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required Kaggle CSV columns: {missing}")

        df = df[df["split"].astype(str).str.lower() == split.lower()].reset_index(drop=True)

        if df.empty:
            raise ValueError(f"No rows found for split={split} in {self.csv_path}")

        self.df = df

        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=7),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["path"]

        image = read_kaggle_image_as_rgb(
            image_path,
            harmonize_preprocess=self.harmonize_preprocess,
            crop_body=self.crop_body,
            percentile_lower=self.percentile_lower,
            percentile_upper=self.percentile_upper,
            crop_threshold=self.crop_threshold,
            crop_margin=self.crop_margin,
        )

        image = self.transform(image)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return {
            "image": image,
            "label": label,
            "path": image_path,
        }