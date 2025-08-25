import base64
import pandas as pd
from datasets import load_dataset
import random
from io import BytesIO
from PIL import Image

# Set seed for reproducibility
random.seed(42)

# Load full dataset
dataset = load_dataset("nanonets/key_information_extraction", split="test")  # it's under "test"

# Shuffle
dataset = dataset.shuffle(seed=42)

# Split proportions
n = len(dataset)
train_split = int(0.8 * n)
val_split = int(0.1 * n)

train_data = dataset[:train_split]
val_data = dataset[train_split:train_split + val_split]
test_data = dataset[train_split + val_split:]

# save the splits to CSV files
train_df = pd.DataFrame(train_data)
val_df = pd.DataFrame(val_data)
test_df = pd.DataFrame(test_data)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)