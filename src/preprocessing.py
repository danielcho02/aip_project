from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def read_dicom_as_rgb(path):
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

    return Image.fromarray(pixels).convert("RGB")


def read_image_as_rgb(path):
    path = Path(path)
    if path.suffix.lower() == ".dcm":
        return read_dicom_as_rgb(path)
    with Image.open(path) as image:
        return image.convert("RGB")


def _otsu_threshold(gray):
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 255))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)

    weight_bg = 0.0
    sum_bg = 0.0
    max_between = -1.0
    threshold = 0

    for level in range(256):
        weight_bg += hist[level]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += level * hist[level]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            threshold = level

    return threshold


def body_crop(image, margin_ratio=0.03, min_area_ratio=0.20):
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    threshold = _otsu_threshold(gray)
    mask = gray > max(5, threshold)

    if mask.mean() < min_area_ratio:
        nonzero = gray > 5
        if nonzero.mean() >= min_area_ratio:
            mask = nonzero

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image

    width, height = image.size
    margin = int(round(max(width, height) * margin_ratio))
    left = max(0, int(xs.min()) - margin)
    right = min(width, int(xs.max()) + margin + 1)
    top = max(0, int(ys.min()) - margin)
    bottom = min(height, int(ys.max()) + margin + 1)

    crop_area = (right - left) * (bottom - top)
    if crop_area < min_area_ratio * width * height:
        return image
    return image.crop((left, top, right, bottom))


def image_to_imagenet_tensor(image, image_size=224):
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor, resized


def preprocess_for_model(image, image_size=224, use_body_crop=True):
    processed = body_crop(image) if use_body_crop else image
    tensor, resized = image_to_imagenet_tensor(processed, image_size=image_size)
    return tensor, resized
