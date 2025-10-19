import wandb
import torch
import io

from PIL import Image
from unsloth import FastVisionModel
from transformers import TextStreamer
from datasets import load_dataset
from unsloth import get_chat_template


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

def load_hf_dataset():
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("nanonets/key_information_extraction")
    return ds['test']

def load_model():
    task_path = "unsloth/gemma-3-4b-pt"

    model, processor = FastVisionModel.from_pretrained(
        model_name=task_path,
        load_in_4bit=False,
        use_gradient_checkpointing = "unsloth",
    )
    FastVisionModel.for_inference(model)
    return model, processor


def process_vqa(model, processor, image, question, device):
    processor = get_chat_template(
        processor,
        "gemma-3"
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    text_streamer = TextStreamer(processor, skip_prompt=True)
    outputs = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.0,
        min_p=0.1,
    )
    return processor.decode(outputs[0], skip_special_tokens=True)


def log_to_wandb(responses, project_name="OCR with Gemma3", run_name="Load 4bit Model"):
    """
    Initialize wandb and log the VQA predictions.
    
    Args:
        responses: List of dictionaries containing image, text, and prediction
        project_name: Name of the wandb project
        run_name: Name of the wandb run
    """
    wandb.init(project=project_name, name=run_name)
    
    # Create table and log predictions
    table = wandb.Table(columns=["image", "text", "prediction"])
    for response in responses:
        table.add_data(wandb.Image(response["image"]), response["text"], response["prediction"])
    
    wandb.log({"VQA Predictions": table})
    wandb.finish()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model()
    dataset = load_hf_dataset()
    responses = []

    for index in range(len(dataset)):
        row = dataset[index]
        image_data = row['image']
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif hasattr(image_data, 'convert'):
            image = image_data.convert("RGB")
        else:
            image = Image.open(image_data).convert("RGB")
            
        prediction = process_vqa(model, processor, image, human_prompt, device)

        responses.append({
            "image": image,
            "text": human_prompt,
            "prediction": prediction
        })
        print(f"[Done] {index} -> {prediction}")

    # log_to_wandb(responses)
