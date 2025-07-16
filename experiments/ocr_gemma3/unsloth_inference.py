import pandas as pd
import wandb
import io
import torch

from PIL import Image
from unsloth import FastVisionModel
from transformers import TextStreamer


system_prompt = """
You are a highly accurate document understanding agent designed to extract structured information from scanned receipts, invoices, and sales slips.

Your goal is to extract a fixed set of predefined fields from a given document image and return them as a single well-formed JSON object.

Follow these rules carefully:
1. Match Labels and Synonyms: Use exact field labels or common variations (e.g., "Tax ID", "GST No.", "TIN").
2. Position Awareness: Use the layout of the document to infer missing labels (e.g., phone number near store name).
3. Text Cleanup: Remove OCR noise, headers, and irrelevant content.
4. Currency Handling: Preserve currency symbols and decimal formatting in monetary values.
5. Missing or Unreadable Fields: If a field is not present or unreadable, return its value as `null`.
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

Return a single JSON object that includes all fields, with "" where necessary.
"""

def load_dataset():
    df = pd.read_csv("nanonets_kie_splits/test.csv")
    return df

def load_model():
    task_path = "unsloth/gemma-3-4b-pt"

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=task_path,
        load_in_4bit=True,
        use_gradient_checkpointing = "unsloth",
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


def process_vqa(model, tokenizer, image, question, device):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    wandb.init(project="OCR with Gemma3", name="Load 4bit Model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model()
    dataset = load_dataset()
    responses = []

    for index, row in dataset.iterrows():
        image_bytes = eval(row['image']) if isinstance(row['image'], str) else row['image']
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prediction = process_vqa(model, tokenizer, image, human_prompt, device)

        responses.append({
            "image": image,
            "text": human_prompt,
            "prediction": prediction
        })
        print(f"[Done] {index} -> {prediction}")

    # Log the image and prediction to wandb
    table = wandb.Table(columns=["image", "text", "prediction"])
    for response in responses:
        table.add_data(wandb.Image(response["image"]), response["text"], response["prediction"])
    
    wandb.log({"VQA Predictions": table})
    wandb.finish()
    wandb.log({"VQA Predictions": table})
