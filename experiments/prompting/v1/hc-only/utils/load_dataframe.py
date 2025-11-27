import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image


def create_image_caption_dataset(
    image_folder: str,
    captions_json: str
) -> pd.DataFrame:
    """
    Reads images and their captions from a JSON (key: filename, value: list of captions)
    and creates a DataFrame with ['image', 'caption'].

    Parameters:
        image_folder: path to folder with image files
        captions_json: path to JSON with image filename → [captions]
        image_size: target size for all images (width, height)
        caption_strategy: either 'first' or 'random' to pick a caption

    Returns:
        A pandas DataFrame with two columns: image (NumPy array) and caption (string)
    """
    image_size = (224, 224)

    with open(captions_json, "r", encoding="utf-8") as f:
        caption_data = json.load(f)  # No "root" key — top-level mapping

    records = []
    for fname in sorted(os.listdir(image_folder)):
        if fname not in caption_data:
            print(f"⚠️ Skipping '{fname}' (no captions found)")
            continue

        caption = caption_data[fname]
        if not caption:
            continue


        image_path = os.path.join(image_folder, fname)
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(image_size, Image.LANCZOS)

        except Exception as e:
            print(f"❌ Failed to load '{fname}': {e}")
            continue

        records.append({
            "image": img,
            "caption": caption
        })

    df = pd.DataFrame(records)
    return df


# def main():
#     image_folder = "/kaggle/input/car-caption-dataset/Captioned_Data/filtered_images"
#     captions_json = "/kaggle/input/car-caption-dataset/Captioned_Data/merged_output.json"

#     df = create_image_caption_dataset(image_folder, captions_json)

#     print(df.head())

#     output_csv = "/kaggle/working/image_caption_data.csv"
#     df[['image', 'caption']].to_csv(output_csv, index=False)
#     print(f"Saved to {output_csv}")


# if __name__ == "__main__":
#     main()