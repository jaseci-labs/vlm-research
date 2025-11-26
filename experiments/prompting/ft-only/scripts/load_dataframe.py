import os
import json
import argparse

from PIL import Image
import numpy as np
import pandas as pd


import os
import json
import pandas as pd
from PIL import Image

def create_image_caption_dataset(
    image_folder: str,
    captions_json: str
) -> pd.DataFrame:
    """
    Reads images and their captions from a JSON (key: filename, value: list or string of captions)
    and creates a DataFrame with ['filename', 'image', 'caption'].

    Parameters:
        image_folder: path to folder with image files
        captions_json: path to JSON with image filename → [captions]

    Returns:
        A pandas DataFrame with three columns:
        - filename: name of the image file
        - image: PIL Image object resized to 224x224
        - caption: string or list of captions from the JSON
    """
    image_size = (224, 224)
    
    # Load caption mapping
    with open(captions_json, "r", encoding="utf-8") as f:
        caption_data = json.load(f)  # Mapping: filename → captions

    records = []
    # Iterate through sorted image files in the folder
    for fname in sorted(os.listdir(image_folder)):
        if fname not in caption_data:
            print(f"⚠️ Skipping '{fname}' (no captions found)")
            continue

        captions = caption_data[fname]
        if not captions:
            print(f"⚠️ Skipping '{fname}' (empty captions)")
            continue

        image_path = os.path.join(image_folder, fname)
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(image_size, Image.LANCZOS)
        except Exception as e:
            print(f"❌ Failed to load '{fname}': {e}")
            continue

        records.append({
            "filename": fname,
            "image": img,
            "caption": captions
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