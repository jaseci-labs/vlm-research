from transformers import AutoProcessor, Gemma3nForConditionalGeneration, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
from datasets import load_dataset
from kie_metrics import evaluate_kie_predictions

import psutil
import wandb
import torch
import time
import io


system_prompt = """
You are a highly accurate document understanding agent designed to extract structured information from scanned receipts, invoices, and sales slips.

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

load_mode = "8bit" 
quant_config = None
model_id = "google/gemma-3-4b-it"

def extract_json_string(text):
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end + 1]
    except:
        pass
    # Return default empty schema
    return '{"date": "", "doc_no_receipt_no": "", "seller_name": "", "seller_address": "", "seller_phone": "", "seller_gst_id": "", "total_tax": "", "total_amount": ""}'


def split_the_dataset(dataset):
    train_temp = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = train_temp['train']
    temp_ds = train_temp['test']

    val_test = temp_ds.train_test_split(test_size=0.5, seed=42)
    val_ds = val_test['train']
    test_ds = val_test['test']
    return train_ds, test_ds, val_ds

preds, gts = [], []
images, inference_times, memory_usages = [], [], []

wandb.init(project="OCR with Gemma3", name="Loading bnb 8bit Model")
table = wandb.Table(columns=["index", "image", "ground_truth", "prediction", "inference_time", "memory_usage", "score"])

process = psutil.Process()

if load_mode == "4bit":
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
elif load_mode == "8bit":
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # llm_int8_threshold=6.0
    )

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quant_config if quant_config else None,
    torch_dtype=torch.bfloat16
).eval()

# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=bnb_config,).eval()
processor = AutoProcessor.from_pretrained(model_id)
dataset = load_dataset("nanonets/key_information_extraction", split="test")
train_ds, test_ds, val_ds = split_the_dataset(dataset)

for i, row in enumerate(test_ds):
    image_bytes = row["image"]
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": human_prompt}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    input_len = inputs["input_ids"].shape[-1]

    torch.cuda.reset_peak_memory_stats()
    # start_reserved = torch.cuda.memory_reserved()
    
    with torch.inference_mode():
        start_time = time.time()
        generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        end_time = time.time()
        generation = generation[0][input_len:]  
        # print(generation)
    
    # peak_reserved = torch.cuda.max_memory_reserved()
    # memory_used = peak_reserved - start_reserved
    # memory_used_MB = memory_used / 1024 / 1024
    peak_vram_bytes = torch.cuda.max_memory_allocated(device="cuda")
    peak_vram_mb = peak_vram_bytes / (1024 ** 2)
    
    print("=" * 30 + "RESPONSE START" + "=" * 30)
    decoded = processor.decode(generation, skip_special_tokens=True)
    json_part = extract_json_string(decoded)
    print("=" * 20 + "Prediction" + "=" * 20)
    print(json_part)

    print("=" * 20 + "Annotations" + "=" * 20)
    print(row["annotations"])
    
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    print(f"Peak Memory usage: {peak_vram_mb} MB")
    print("=" * 30 + "RESPONSE END" + "=" * 30)
    print()

    preds.append(json_part)
    gts.append(row["annotations"])
    images.append(image)
    inference_times.append(end_time - start_time)
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