import json
import os

def normalize_caption_format(json_path, output_path=None):
    if output_path is None:
        base, ext = os.path.splitext(json_path)
        output_path = f"{base}_normalized{ext}"

    with open(json_path, "r") as f:
        data = json.load(f)

    updated_data = {}
    for image_path, content in data.items():
        visual_scene = content.get("visual_scene", {})
        caption = visual_scene.pop("caption", None)

        updated_data[image_path] = {
            "visual_scene": visual_scene,
            "caption": caption
        }

    with open(output_path, "w") as f:
        json.dump(updated_data, f, indent=2)

    print(f"Normalized JSON saved to: {output_path}")

if __name__ == "__main__":
    json_path = "../image_dataset/res.json"
    normalize_caption_format(json_path)
