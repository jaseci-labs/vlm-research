import requests
import zipfile
import os

# URL of the zip file
url = 'https://dms.uom.lk/s/xspcQEbgAgkK6P3/download'

# Local filenames
zip_filename = 'downloaded_file.zip'
extract_dir = '.'

# Step 1: Download the zip file
print("Downloading...")
response = requests.get(url)
with open(zip_filename, 'wb') as f:
    f.write(response.content)
print("Download complete.")

# Step 2: Unzip the file
print("Extracting...")
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f"Extracted to '{extract_dir}'")

# Optional: Clean up the zip file
# os.remove(zip_filename)
