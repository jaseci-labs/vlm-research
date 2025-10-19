from together import Together

import base64
import random
import json
import time
import os

ISSUE_PATH = "/home/malitha/programming/vlm-research/experiments/knowledge_distillation/phase_3/model_performance_comparison/problem.txt"

IMAGES_PATH = "/home/malitha/programming/vlm-research/experiments/knowledge_distillation/phase_3/image_dataset/imagepaths.txt"
JSON_PATH = "/home/malitha/programming/vlm-research/experiments/knowledge_distillation/phase_3/model_performance_comparison/res.json"
API_KEYS_PATH = "together_api_keys.txt"

MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"

with open(API_KEYS_PATH, "r") as f:
    api_keys = [line.strip() for line in f if line.strip()]

def image_to_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

def response(image_url, client):
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a remote sensing image captioning assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate accurate, concise caption describing key visual elements like land use, objects, or spatial patterns in the provided image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_base64(image_url)
                        }
                    }
                ]
            }
        ]
    )
    if completion.choices is None:
        print("Quota end")
        return None
    out = completion.choices[0].message.content
    return out

if api_keys:
    os.environ["TOGETHER_API_KEY"] = api_keys[0]
    client = Together()
else:
    client = None

try:
    with open(JSON_PATH, "r") as f:
        res_file = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    res_file = {}

with open(IMAGES_PATH, "r") as f:
    lines = [line.strip() for line in f]

for i in range(len(lines)):
    image_path = "../" + lines[i]
    success = False
    attempts = 0

    while not success and attempts < 5:
        if client is None:
            print("No API key available.")
            break

        print(f"Processing {image_path} (Attempt {attempts + 1})")
        attempts += 1
        try:
            res = response(image_path, client)
            time.sleep(random.uniform(5, 8))

            if res is None:
                raise Exception("Quota or model error")

            # Save the raw response string directly
            res_file[lines[i]] = res
            with open(JSON_PATH, "w") as f:
                json.dump(res_file, f, indent=2)
            print(f"Saved result for {lines[i]}")
            success = True

        except Exception as e:
            print(f"Error: {e}")
            with open(ISSUE_PATH, "a") as f:
                f.write(lines[i] + "\n")
            break