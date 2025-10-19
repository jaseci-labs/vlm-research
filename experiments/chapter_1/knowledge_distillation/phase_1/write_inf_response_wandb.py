from PIL import Image
from unsloth import FastVisionModel
from transformers import TextStreamer
from huggingface_hub import login
import wandb
import os

def load_model(task_path):
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
        max_new_tokens=500,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    login(token="hf_XXXXXXXXXXXXXXXX")

    wandb.init(
        project="Gemma 3 4B Distillation",
        name="Gemma-finetuned-vs-base-comparison"
    )

    # Load both models
    finetuned_model, finetuned_tokenizer = load_model("Warun/Jaseci-Gemma-3-4B-Unsloth")
    base_model, base_tokenizer = load_model("unsloth/gemma-3-4b-it")

    question = "Describe the damages of the car."
    table = wandb.Table(columns=["image_name", "image", "text", "finetuned_prediction", "baseline_prediction"])

    with open("test_images/test_images_paths.txt", "r") as f:
        image_names = [line.strip() for line in f if line.strip()]

    for img_name in image_names:
        img_path = img_name
        if not os.path.exists(img_path):
            print(f"[Warning] Skipping missing image: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")

        pred_finetuned = process_vqa(finetuned_model, finetuned_tokenizer, image, question)
        pred_base = process_vqa(base_model, base_tokenizer, image, question)

        print(f"[Done] {img_name}\n  Finetuned: {pred_finetuned}\n  Base: {pred_base}")

        # Add both to W&B
        table.add_data(
            img_name,
            wandb.Image(image),
            question,
            pred_finetuned,
            pred_base
        )

    wandb.log({"VQA Comparison Table": table})
