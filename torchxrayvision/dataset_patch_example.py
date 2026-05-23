"""
Patch example for src/dataset.py.

Your KaggleChestXrayDataset already exists, so do not blindly replace the whole file
unless its interface is the same. Apply the changes below:

1. Add this import near the top:
    try:
        from xray_preprocess import read_kaggle_image_as_rgb
    except ImportError:
        from src.xray_preprocess import read_kaggle_image_as_rgb

2. Add these arguments to KaggleChestXrayDataset.__init__:
    harmonize_preprocess=True,
    crop_body=True,
    percentile_lower=1.0,
    percentile_upper=99.0,
    crop_threshold=5,
    crop_margin=10,

3. Store them:
    self.harmonize_preprocess = harmonize_preprocess
    self.crop_body = crop_body
    self.percentile_lower = percentile_lower
    self.percentile_upper = percentile_upper
    self.crop_threshold = crop_threshold
    self.crop_margin = crop_margin

4. Replace the image read line in __getitem__.

Before:
    image = Image.open(image_path).convert("RGB")

After:
    image = read_kaggle_image_as_rgb(
        image_path,
        do_percentile=self.harmonize_preprocess,
        do_crop=self.crop_body,
        lower=self.percentile_lower,
        upper=self.percentile_upper,
        crop_threshold=self.crop_threshold,
        crop_margin=self.crop_margin,
    )

The rest of the Dataset class, including transforms and return dict, can remain unchanged.
"""
