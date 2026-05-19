from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class KaggleChestXrayDataset(Dataset):
    def __init__(self, csv_path, split, image_size=224, augment=False):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'.")

        self.csv_path = Path(csv_path)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        df = pd.read_csv(self.csv_path)
        required_columns = {"path", "label", "class_name", "group", "split"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV columns: {missing}")

        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split='{split}' in {self.csv_path}")

        transform_steps = []
        if split == "train" and augment:
            transform_steps.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=10),
                ]
            )

        transform_steps.extend(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.transform = transforms.Compose(transform_steps)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["path"]
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "path": image_path,
        }
