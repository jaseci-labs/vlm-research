import json
import os
import wandb
from PIL import Image

wandb.init(project="Model Performance Comparison", name="Model Performance Comparison")
JSON_PATH = "/home/malitha/programming/vlm-research/experiments/knowledge_distillation/phase_3/model_performance_comparison/res.json"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

table = wandb.Table(columns=["image_path", "image", "teacher_response"])

image_base_path = "../"

for image_path, metadata in data.items():
    full_path = os.path.join(image_base_path, image_path)
    
    if not os.path.exists(full_path):
        print(f"Image not found: {full_path}")
        continue

    img = Image.open(full_path)

    table.add_data(
        image_path,
        wandb.Image(img, caption=image_path),
        json.dumps(metadata, indent=2)
    )

wandb.log({"Image JSON Table": table})

wandb.finish()
