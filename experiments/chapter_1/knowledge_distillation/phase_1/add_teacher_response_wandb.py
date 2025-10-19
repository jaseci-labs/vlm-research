import wandb
import json
import os

# Load the JSON file
with open("image_dataset/test.json", "r") as f:
    teacher_data = json.load(f)

# Prepare a mapping from filename to the combined teacher response string
filename_to_teacher_response = {}
for full_path, entry in teacher_data.items():
    filename = os.path.basename(full_path)
    # Combine predictions and report into one dictionary
    combined_info = {
        "predictions": entry.get("predictions", []),
        "report": entry.get("report", "No report available")
    }
    # Convert to a nicely formatted JSON string
    teacher_response_str = json.dumps(combined_info, indent=2)
    filename_to_teacher_response[filename] = teacher_response_str

# Initialize W&B API and get artifact & table
api = wandb.Api()
artifact = api.artifact("vlm-research/Gemma-3-4B-Distillation/run-kh2or85n-VQAComparisonTable:v0")
table = artifact.get("VQA Comparison Table")  # confirm key name

# Add a new column with teacher responses
new_columns = table.columns + ["teacher_response"]
new_data = []

for row in table.data:
    row_dict = dict(zip(table.columns, row))
    image_path = row_dict.get("image_name", "")
    filename = os.path.basename(image_path)

    teacher_response = filename_to_teacher_response.get(filename, "No teacher response available")
    new_data.append(row + [teacher_response])

# Create new W&B table and log it
updated_table = wandb.Table(columns=new_columns, data=new_data)

wandb.init(project="Gemma-3-4B-Distillation", name="updated_with_teacher_response", entity="vlm-research")
wandb.log({"table_with_teacher_response": updated_table})
wandb.finish()
