from openai import OpenAI

import base64
import random
import json
import time
import re

# FIle paths
ISSUE_PATH = "/home/malitha/programming/vlm-research/experiments/knowledge_distillation/phase_3/image_dataset/problem.txt"

# All Images should be inside image_dataset folder
IMAGES_PATH = "/home/malitha/programming/vlm-research/experiments/knowledge_distillation/phase_3/image_dataset/imagepaths.txt"
JSON_PATH = "/home/malitha/programming/vlm-research/experiments/knowledge_distillation/phase_3/image_dataset/res.json"
API_KEYS_PATH = "api_keys.txt"

# Model and API configuration
MODEL_NAME = "qwen/qwen2.5-vl-72b-instruct:free"
API_URL = "https://openrouter.ai/api/v1"


# Read API keys from the txt file
with open(API_KEYS_PATH, "r") as f:
    api_keys = [line.strip() for line in f if line.strip()]
api_keys_index = 0


def get_new_client(api_key):
    return OpenAI(
        base_url=API_URL,
        api_key=api_key,
    )

def get_next_api_key():
    global api_keys_index
    if api_keys_index < len(api_keys):
        api_key = api_keys[api_keys_index]
        api_keys_index += 1
        return api_key
    else:
        print("All API keys exhausted!")
        return None
    

def image_to_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"


# Function to get the response from the model
def response(image_url, client):
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in remote sensing image analysis and captioning. "
                    "Given a remote sensing image, analyze the scene and provide a detailed "
                    "structured JSON response including: the overall scene type (e.g., airport, urban area, forest), "
                    "key visual elements present in the scene, a list of dominant objects with their type, count, "
                    "and descriptive attributes, and a concise but descriptive caption summarizing the scene.\n\n"
                    "For each dominant object's description, provide a short, visually grounded summary including relevant details "
                    "such as spatial or positional information (e.g., \"clustered near the river\"), physical characteristics "
                    "(e.g., \"large rectangular buildings\"), function if inferable (e.g., \"used for cargo storage\"), or visible condition "
                    "(e.g., \"under construction\"). Do not repeat count information in the description, as it is provided separately.\n\n"
                    "If it is hard to get the exact count of the objects, use approximate terms such as \"many,\" \"multiple,\" or similar measurements.\n\n"
                    "Format the response exactly as a JSON object like this:\n"
                    "```json\n"
                    "{\n"
                    "  \"visual_scene\": {\n"
                    "    \"scene_type\": \"<scene_type>\",\n"
                    "    \"key_elements\": [\n"
                    "      \"<element_1>\",\n"
                    "      \"<element_2>\",\n"
                    "      \"...\"\n"
                    "    ],\n"
                    "    \"dominant_objects\": [\n"
                    "      {\n"
                    "        \"type\": \"<object_type>\",\n"
                    "        \"attributes\": {\n"
                    "          \"count\": \"<exact_or_approximate_count>\",\n"
                    "          \"description\": \"<spatial_position_appearance_function_condition>\"\n"
                    "        }\n"
                    "      }\n"
                    "    ],\n"
                    "    \"caption\": \"<concise_descriptive_caption>\"\n"
                    "  }\n"
                    "}\n"
                    "```\n"
                    "Provide only the JSON response without any extra commentary."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze the following remote sensing image and provide the response in the specified JSON format."
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


# Initialize
current_api_key = get_next_api_key()
client = get_new_client(current_api_key) if current_api_key else None

# Load existing results
try:
    with open(JSON_PATH, "r") as f:
        res_file = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    res_file = {}

# Read image paths
with open(IMAGES_PATH, "r") as f:
    lines = [line.strip() for line in f]

# Main loop
for i in range(37, len(lines)):
    image_path = lines[i]
    success = False
    attempts = 0

    while not success and attempts < 5:
        if client is None:
            print("No more API keys available.")
            break

        print(f"Processing {image_path} (Attempt {attempts + 1})")
        attempts += 1
        try:
            res = response(image_path, client)
            time.sleep(random.uniform(5, 8))

            if res is None:
                raise Exception("Quota or model error")

            match = re.search(r'{.*}', res, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                success = True
                res_file[image_path] = data
                with open(JSON_PATH, "w") as f:
                    json.dump(res_file, f, indent=2)
                print(f"Saved result for {image_path}")
            else:
                print("No valid JSON structure found.")

        except Exception as e:
            print(f"Error: {e}")
            # Attempt to switch API key
            current_api_key = get_next_api_key()
            if current_api_key:
                client = get_new_client(current_api_key)
                print("Switched to new API key.")
            else:
                print("All API keys exhausted.")
                with open(ISSUE_PATH, "a") as f:
                    f.write(image_path + "\n")
                break