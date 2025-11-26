from openai import OpenAI, OpenAIError, RateLimitError
import base64
import wandb
import time
import json

# Load API keys from file
with open("openrouter_keys.txt", "r") as f:
    api_keys = [line.strip() for line in f if line.strip()]

api_index = 0

client = OpenAI(
    api_key=api_keys[api_index],
    base_url="https://openrouter.ai/api/v1"
)

MODEL_NAME = "openai/gpt-4o-mini"

wandb.login()
run = wandb.init(project="Gemma 3 4B Distillation Phase 2", name="AI_evaluated_score", entity="vlm-research")

artifact = wandb.use_artifact("vlm-research/Gemma 3 4B Distillation Phase 2/run-e0obv6yk-VQAComparisonTable:v0")
table = artifact.get("VQA Comparison Table")

new_columns = table.columns + ["evaluation_scores"]
new_table = wandb.Table(columns=new_columns)

def switch_api_key():
    global api_index, client
    api_index += 1
    if api_index >= len(api_keys):
        raise Exception("All API keys exhausted.")
    client = OpenAI(
        api_key=api_keys[api_index],
        base_url="https://openrouter.ai/api/v1"
    )
    print(f"Switched to new API key: Index {api_index}")

def image_to_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

def call_openrouter(prompt, image_name):
    image_data = image_to_base64(image_name)

    for attempt in range(5):  # Retry up to 5 times
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Evaluate the student and teacher responses to the car damage image as JSON scores only."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]}
                ]
            )
            return response.choices[0].message.content  # Extract the message content
        except RateLimitError as e:
            print(f"RateLimitError: {e} - Switching API key.")
            try:
                switch_api_key()
            except Exception as ex:
                print(f"All keys exhausted. Exiting: {ex}")
                raise
        except OpenAIError as e:
            print(f"OpenAIError: {e} - Retrying after backoff.")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    return None

for row in table.data:
    image_name = row[0]
    teacher_output = row[5]
    student_untrained = row[4]
    student_trained = row[3]

    prompt = f"""
Evaluate the following outputs from a teacher model, a student model without training, and a student model with training, in the context of the provided car damage image...

Teacher Model Output:
{teacher_output}

Student Model Without Training Output:
{student_untrained}

Student Model With Training Output:
{student_trained}

Evaluation Tasks:
Accuracy Scores:
Assess how accurately each model (teacher, student without training, student with training) describes the damages shown in the image.
Assign a score (1-10) for each model's accuracy based on completeness and correctness.
Replication Scores:
For the student models (without training and with training), evaluate how well they replicate the teacher model's output in terms of:
Format (e.g., structured JSON vs. unstructured text).
Content (e.g., damage details and report phrasing).
Assign a score (1-10) for each student model's replication of the teacher model.
Trained vs. Untrained Student Comparison:
Compare the accuracy of the student model with training to the student model without training, focusing only on how well each describes the damages (ignore structure differences).
Indicate which is more accurate and by how much (e.g., based on the accuracy scores).
Please provide scores (1-10) for each task and a brief explanation for each score. (edited)

Provide only JSON formatted evaluation scores as:
{{
  "accuracy_scores": {{
    "teacher_model": int,
    "student_model_untrained": int,
    "student_model_trained": int
  }},
  "replication_scores": {{
    "student_model_untrained": {{
      "format": int,
      "content": int
    }},
    "student_model_trained": {{
      "format": int,
      "content": int
    }}
  }},
  "trained_vs_untrained_comparison": {{
    "more_accurate_model": str,
    "accuracy_difference": int
  }}
}}
    """.strip()

    try:
        eval_response = call_openrouter(prompt, image_name)
        print(f"Raw response for {image_name}:\n{eval_response}")
    except Exception as e:
        print(f"Failed to process {image_name}: {e}")
        eval_response = "{}"

    new_row = row + [eval_response]
    new_table.add_data(*new_row)

# Save results
run.log({"evaluation_results": new_table})
run.finish()
