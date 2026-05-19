from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RSNAPneumoniaDataset(Dataset):
    def __init__(self, rsna_root, sample_per_class=None, image_size=224, seed=42):
        self.rsna_root = Path(rsna_root)
        self.image_dir = self.rsna_root / "stage_2_train_images"
        labels_csv = self.rsna_root / "stage_2_train_labels.csv"

        if not labels_csv.is_file():
            raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Missing DICOM directory: {self.image_dir}")

        labels_df = pd.read_csv(labels_csv)
        required_columns = {"patientId", "Target"}
        missing_columns = required_columns - set(labels_df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required RSNA columns: {missing}")

        df = (
            labels_df.groupby("patientId", as_index=False)["Target"]
            .max()
            .rename(columns={"patientId": "patient_id", "Target": "label"})
        )
        df["path"] = df["patient_id"].map(lambda patient_id: str(self.image_dir / f"{patient_id}.dcm"))

        if sample_per_class is not None:
            if sample_per_class <= 0:
                raise ValueError("sample_per_class must be a positive integer or None.")
            sampled = []
            for label in (0, 1):
                label_df = df[df["label"] == label]
                n_samples = min(sample_per_class, len(label_df))
                sampled.append(label_df.sample(n=n_samples, random_state=seed))
            df = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

        self.df = df.reset_index(drop=True)
        if self.df.empty:
            raise ValueError("No RSNA samples found.")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.df)

    def _read_dicom_as_rgb(self, path):
        dicom = pydicom.dcmread(path)
        pixels = dicom.pixel_array.astype(np.float32)

        photometric = str(getattr(dicom, "PhotometricInterpretation", "")).strip().upper()
        if photometric == "MONOCHROME1":
            pixels = pixels.max() - pixels

        min_value = pixels.min()
        max_value = pixels.max()
        if max_value > min_value:
            pixels = (pixels - min_value) / (max_value - min_value)
            pixels = (pixels * 255.0).clip(0, 255).astype(np.uint8)
        else:
            pixels = np.zeros_like(pixels, dtype=np.uint8)

        image = Image.fromarray(pixels)
        return image.convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["path"]
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        image = self._read_dicom_as_rgb(image_path)
        image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "patient_id": row["patient_id"],
            "path": image_path,
        }
