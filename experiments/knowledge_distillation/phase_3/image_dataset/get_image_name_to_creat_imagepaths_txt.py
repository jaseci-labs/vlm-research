import os

# Path to the folder containing your images
image_folder = "rsicd_images"
output_txt_file = "imagepaths.txt"

# Get all .jpg filenames in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

# Sort the list for consistency (optional)
image_files.sort()

# Create the formatted lines
formatted_lines = [f"image_dataset/{image_folder}/{img}" for img in image_files]

# Write to the txt file
with open(output_txt_file, "w") as f:
    for line in formatted_lines:
        f.write(line + "\n")

print(f"Saved {len(formatted_lines)} image paths to {output_txt_file}")
