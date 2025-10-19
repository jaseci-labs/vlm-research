import csv
import os
import sys
import ast
from io import BytesIO
from PIL import Image

csv.field_size_limit(sys.maxsize)

# Paths
csv_file_path = '/home/malitha/programming/vlm-research/experiments/knowledge_distillation/phase_3/select_data_rows/random_3000_rows.csv'
output_folder = '../image_dataset'
bad_log_path = 'corrupt_images.txt'

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

bad_images = []

with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) < 3:
            continue

        image_name = row[0].strip()
        image_bytes_field = row[2].strip()

        print(image_name)

        try:
            # Safely parse string to dict
            parsed_dict = ast.literal_eval(image_bytes_field)

            # Get raw bytes from 'bytes' key
            image_bytes = parsed_dict['bytes']
            image = Image.open(BytesIO(image_bytes))
            image.verify()
            image = Image.open(BytesIO(image_bytes))
            image_path = os.path.join(output_folder, image_name)
            image.save(image_path)
            print(f'Saved: {image_path}')

        except Exception as e:
            print(f'Failed to save {image_name}: {e}')
            bad_images.append(image_name)

# Log corrupt images
if bad_images:
    with open(bad_log_path, 'w') as f:
        for name in bad_images:
            f.write(name + '\n')
    print(f"\nLogged {len(bad_images)} failed images to {bad_log_path}")
