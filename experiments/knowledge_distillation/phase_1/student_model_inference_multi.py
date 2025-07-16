from PIL import Image
from unsloth import FastVisionModel
from transformers import TextStreamer
from huggingface_hub import login
import wandb
import os

def load_model():
    task_path = "Warun/Jaseci-Gemma-3-4B-Unsloth" # put the output model path here

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=task_path,
        load_in_4bit=False,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


def process_vqa(model, tokenizer, image, question):
    messages = [
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
    ).to("cuda")

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
    # Login to Hugging Face Hub and wandb
    login(token="hf_XXXXXXXXXXXXXXXX")
    wandb.init(project="vlm-inference", name="batch-vqa-run")
    
    model, tokenizer = load_model()
    
    responses = []
    question = "Describe the damages of the car."

    with open("test_images/test_images_paths.txt", "r") as f:
        image_names = [line.strip() for line in f if line.strip()]
    
    for img_name in image_names:
        img_path = os.path.join("test_images", img_name)
        if not os.path.exists(img_path):
            print(f"[Warning] Skipping missing image: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        prediction = process_vqa(model, tokenizer, image, question)

        responses.append({
            "image": image,
            "text": question,
            "prediction": prediction
        })
        print(f"[Done] {img_name} -> {prediction}")

    # Log the image and prediction to wandb
    table = wandb.Table(columns=["image", "text", "prediction"])
    for response in responses:
        table.add_data(wandb.Image(response["image"]), response["text"], response["prediction"])
    
    wandb.log({"VQA Predictions": table})
