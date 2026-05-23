import numpy as np
from PIL import Image


def percentile_normalize(arr, lower=1.0, upper=99.0, eps=1e-6):
    """
    X-ray intensity를 percentile clipping으로 정규화.
    Kaggle JPEG와 RSNA DICOM 모두 같은 방식으로 처리하기 위한 함수.
    """
    arr = np.asarray(arr).astype(np.float32)

    low = np.percentile(arr, lower)
    high = np.percentile(arr, upper)

    if high <= low:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low + eps)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    return arr


def crop_non_black_area(arr, threshold=5, margin=10):
    """
    검은 배경을 제거하기 위한 body/background crop.

    arr:
        uint8 grayscale image.
    threshold:
        non-background로 볼 pixel 기준.
    margin:
        crop할 때 주변에 남길 여백.
    """
    arr = np.asarray(arr)

    mask = arr > threshold
    if mask.sum() == 0:
        return arr

    ys, xs = np.where(mask)

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())

    h, w = arr.shape[:2]

    y1 = max(0, y1 - margin)
    y2 = min(h - 1, y2 + margin)
    x1 = max(0, x1 - margin)
    x2 = min(w - 1, x2 + margin)

    return arr[y1:y2 + 1, x1:x2 + 1]


def preprocess_xray_array(
    arr,
    harmonize_preprocess=True,
    crop_body=True,
    percentile_lower=1.0,
    percentile_upper=99.0,
    crop_threshold=5,
    crop_margin=10,
):
    """
    Kaggle / RSNA 공통 X-ray preprocessing.

    1. grayscale array 입력
    2. percentile clipping 또는 min-max normalization
    3. body crop
    4. PIL RGB 이미지 반환
    """
    arr = np.asarray(arr).astype(np.float32)

    if harmonize_preprocess:
        arr = percentile_normalize(
            arr,
            lower=percentile_lower,
            upper=percentile_upper,
        )
    else:
        min_value = arr.min()
        max_value = arr.max()

        if max_value > min_value:
            arr = (arr - min_value) / (max_value - min_value)
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

    if crop_body:
        arr = crop_non_black_area(
            arr,
            threshold=crop_threshold,
            margin=crop_margin,
        )

    image = Image.fromarray(arr).convert("RGB")
    return image


def read_kaggle_image_as_rgb(
    path,
    harmonize_preprocess=True,
    crop_body=True,
    percentile_lower=1.0,
    percentile_upper=99.0,
    crop_threshold=5,
    crop_margin=10,
):
    """
    Kaggle JPEG/PNG 이미지를 읽고 RSNA와 동일한 전처리를 적용.
    """
    image = Image.open(path).convert("L")
    arr = np.asarray(image).astype(np.float32)

    image = preprocess_xray_array(
        arr,
        harmonize_preprocess=harmonize_preprocess,
        crop_body=crop_body,
        percentile_lower=percentile_lower,
        percentile_upper=percentile_upper,
        crop_threshold=crop_threshold,
        crop_margin=crop_margin,
    )

    return image