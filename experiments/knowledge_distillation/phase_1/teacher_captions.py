from openai import OpenAI
import base64
import random
import json
import time
import re

# FIle paths
ISSUE_PATH = "/home/malithaprabhashana/programming/intern/vlm-project/research/distillation/image_dataset/prog/problem.txt"

# All Images should be inside image_dataset folder
IMAGES_PATH = "/home/malithaprabhashana/programming/intern/vlm-project/research/distillation/image_dataset/imagepaths.txt"
JSON_PATH = "/home/malithaprabhashana/programming/intern/vlm-project/research/distillation/image_dataset/res.json"
API_KEYS_PATH = "api_keys.txt"

# Model and API configuration
MODEL_NAME = "qwen/qwen2.5-vl-72b-instruct:free"
API_URL = "https://openrouter.ai/api/v1"


# Read API keys from the txt file
with open(API_KEYS_PATH, "r") as f:
    api_keys = f.readlines()

api_keys = [key.strip() for key in api_keys]

def get_new_client(api_key):
    return OpenAI(
        base_url=API_URL,
        api_key=api_key,
    )

def image_to_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

# Function to get the response from the model
def response(image_url, client):
  completion = client.chat.completions.create(
    model = MODEL_NAME,
    messages=[
      {
        "role": "system",
        "content": "You are an expert in analyzing car damage and writing insurance reports based on provided images and descriptions."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Please provide response based on the provided output format. Expected output format:\n```json\n{\n  \"predictions\": [\n    {\n      \"location\": \"front bumper\",\n      \"damage_type\": \"dent\",\n      \"severity\": \"major\"\n    },\n    {\n      \"location\": \"driver side door\",\n      \"damage_type\": \"scratch\",\n      \"severity\": \"minor\"\n    }\n  ],\n \"report\":\"Insurance Report: The vehicle sustained significant damage, including a major dent on the front bumper and a minor scratch on the driver side door. Estimated repair cost: $1,500.\"}\n```"
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

def get_next_api_key():
    global api_keys_index
    if api_keys_index < len(api_keys):
        api_key = api_keys[api_keys_index]
        api_keys_index += 1
        return api_key
    else:
        print("All API keys exhausted!")
        return None


# Initialize the OpenAI client with the first API key
api_keys_index = 0
current_api_key = get_next_api_key()
client = None
quota_end = False

# Load existing results if the file exists
try:
    with open(JSON_PATH, "r") as f:
        res_file = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    res_file = {}


if current_api_key:
    client = get_new_client(current_api_key)

with open(IMAGES_PATH, "r") as f:
    lines = f.readlines()

for i in range(len(lines)):
    lines[i] = lines[i].strip()

for i in range(1, 1001):
    success = False
    n = 0
    while not success and n < 5 and not quota_end:
        n += 1
        res = response(lines[i], client)
        time.sleep(random.uniform(5, 8))
        
        if res is None:
            print("Quota likely exhausted or API error.")
            quota_end = True
            break
        
        print(f"Attempt {n} raw response:\n{res}\n")
        
        match = re.search(r'{.*}', res, re.DOTALL)
        print(f"Attempt {n} regex match:\n{match}\n")
        if match:
            success = True
            try:
                json_str = match.group(0)
                data = json.loads(json_str)
                success = True
            except json.JSONDecodeError:
                print("JSON decoding error")
                continue

    if not success and not quota_end:
        print("Failed to get a valid response after 5 attempts")
        with open(ISSUE_PATH, "a") as f:
            f.write(lines[i] + "\n")
        continue

    if quota_end:
        print("Quota end, switching API key")
        current_api_key = get_next_api_key()
        if current_api_key:
            client = get_new_client(current_api_key)
            quota_end = False
            continue
        else:
            print("All API keys exhausted, stopping the process.")
            break

    print(data)

    res_file[lines[i]] = data
    with open(JSON_PATH, "w") as f:
        json.dump(res_file, f, indent=2)
    print(f"Saved result for {lines[i]} to JSON file.")
