import io
import time
import torch
import wandb

from PIL import Image
from unsloth import FastVisionModel
from huggingface_hub import login
from datasets import load_dataset
from kie_metrics import evaluate_kie_predictions
system_prompt = """
You are a highly accurate document understanding agent 
designed to extract structured information from scanned receipts, invoices, and sales slips.

Your goal is to extract a fixed set of predefined fields from a given document image and return them as a single well-formed JSON object.

Follow these rules carefully:
1. Match Labels and Synonyms: Use exact field labels or common variations (e.g., "Tax ID", "GST No.", "TIN").
2. Position Awareness: Use the layout of the document to infer missing labels (e.g., phone number near store name).
3. Text Cleanup: Remove OCR noise, headers, and irrelevant content.
4. Currency Handling: Preserve currency symbols and decimal formatting in monetary values.
5. Missing or Unreadable Fields: If a field is not present or unreadable, return its value as empty string.
6. Field Consistency: Always return the same 8 fields, in the exact order shown below.

Output format must strictly match this schema:
{
  "date": "DD/MM/YYYY or similar format",
  "doc_no_receipt_no": "...",
  "seller_name": "...",
  "seller_address": "...",
  "seller_phone": "...",
  "seller_gst_id": "...",
  "total_tax": "...",
  "total_amount": "..."
}
"""

human_prompt = """
Extract the following fields from the given document image:
- date
- doc_no_receipt_no
- seller_name
- seller_address
- seller_phone
- seller_gst_id
- total_tax
- total_amount

Return a single JSON object that includes all fields, with empty string where necessary.
"""


def extract_json_string(text):
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end + 1]
    except:
        pass
    return '{"date": "", "doc_no_receipt_no": "", "seller_name": "", "seller_address": "", "seller_phone": "", "seller_gst_id": "", "total_tax": "", "total_amount": ""}'


def split_the_dataset(dataset):
    train_temp = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = train_temp['train']
    temp_ds = train_temp['test']
    val_test = temp_ds.train_test_split(test_size=0.5, seed=42)
    val_ds = val_test['train']
    test_ds = val_test['test']
    return train_ds, test_ds, val_ds


def load_model():
    task_path = "Warun/Gemma3-4bit-OCR-Unsloth-5-epochs"
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=task_path,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


def process_vqa(model, tokenizer, image, question, device):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to(device)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )
    end_time = time.time()

    peak_vram_bytes = torch.cuda.max_memory_allocated(device="cuda")
    peak_vram_mb = peak_vram_bytes / 1024 / 1024
    inference_time = end_time - start_time

    return tokenizer.decode(outputs[0], skip_special_tokens=True), inference_time, peak_vram_mb


if __name__ == "__main__":
    login(token="hf_xxxxxxxxxxxxxxxxxxxxx")

    preds, gts = [], []
    images, inference_times, memory_usages = [], [], []

    wandb.init(project="OCR with Gemma3", name="Load 4bit Finetuned Model (Final)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model()

    dataset = load_dataset("nanonets/key_information_extraction", split="test")
    train_ds, test_ds, val_ds = split_the_dataset(dataset)

    table = wandb.Table(columns=["index", "image", "ground_truth", "prediction", "inference_time", "memory_usage", "score"])

    for i, row in enumerate(test_ds):
        image_bytes = row["image"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prediction, inference_time, peak_vram_mb = process_vqa(model, tokenizer, image, human_prompt, device)
        json_part = extract_json_string(prediction)

        print("=" * 20 + f" Prediction {i} " + "=" * 20)
        print(json_part)
        print("Ground truth:", row["annotations"])
        print(f"Inference time: {inference_time:.2f} seconds")
        print(f"Peak VRAM: {peak_vram_mb:.2f} MB")
        print("=" * 60)

        preds.append(json_part)
        gts.append(row["annotations"])
        images.append(image)
        inference_times.append(inference_time)
        memory_usages.append(peak_vram_mb)

    avg_score, all_scores = evaluate_kie_predictions(preds, gts)
    print("avg_score", avg_score)
    print("all_scores", all_scores)

    lists = [preds, gts, images, inference_times, memory_usages, all_scores]
    lengths = [len(lst) for lst in lists]
    
    if len(set(lengths)) != 1:
        raise ValueError(f"Mismatch in list lengths: {lengths}")

    for i in range(len(preds)):
        table.add_data(
            i,
            wandb.Image(images[i]),
            str(gts[i]),
            preds[i],
            inference_times[i],
            int(memory_usages[i]),
            all_scores[i],
        )

    wandb.log({"Predictions Table": table, "Average Score": avg_score})
    wandb.finish()
