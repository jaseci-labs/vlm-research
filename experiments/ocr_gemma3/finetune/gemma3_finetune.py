from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
# from unsloth import is_bf16_supported
# from transformers import EarlyStoppingCallback

from datasets import load_dataset
from huggingface_hub import login
from PIL import Image

# import wandb
import torch
import json
import io

wandb.init(
    project="OCR with Gemma3",
    name="Finetuning 4bit Unsloth",
    group="gemma tests",
    tags=["gemma","vision", "finetune"],
    notes="Testing gemma 3 4B Unsloth",
    config={
        "model": "Gemma 3 4B",
    },
)

MODEL_NAME = "unsloth/gemma-3-4b-it"

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

def split_the_dataset(dataset):
    train_temp = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = train_temp['train']
    temp_ds = train_temp['test']

    val_test = temp_ds.train_test_split(test_size=0.5, seed=42)
    val_ds = val_test['train']
    test_ds = val_test['test']
    return train_ds, test_ds, val_ds


def configure_model(MODEL_NAME):
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 16,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 16,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )
    return model, tokenizer


def configuration_for_training(model, tokenizer, train_data, val_data, epochs=1):
    FastVisionModel.for_training(model)

    # callbacks = [
    # EarlyStoppingCallback(early_stopping_patience=3),
    # ]
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = train_data,
        eval_dataset = val_data,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # max_steps = 30,
            num_train_epochs = epochs, # Set this instead of max_steps for full training runs
            learning_rate = 2e-4,
            fp16 = False,
            bf16 = True,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            # report_to = "wandb",
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 2048,
            eval_strategy="steps",
            eval_steps=2,
    ),
    # load_best_model_at_end=True,
    # metric_for_best_model="eval_loss",
    # callbacks=callbacks,
    )
    return trainer


def convert_to_conversation(sample):
    image_bytes = sample["image"]
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": human_prompt},
                {"type": "image", "image": image},
            ],
        },
        {
            "role": "assistant", 
            "content": [{"type": "text", "text": json.dumps(sample["annotations"])}]
        },
    ]
    return {"messages": conversation}


def save_finetuned_model(model, tokenizer, model_name):
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)


def upload_to_huggingface_hub(model, processor):
    model.push_to_hub("Warun/Gemma3-4bit-OCR-Unsloth")
    processor.push_to_hub("Warun/Gemma3-4bit-OCR-Unsloth")


# Main execution
if __name__ == "__main__":
    login(token="hf_xxxxxxxxxxxxxxxxxxxxx")
    preds, gts = [], []
    images, inference_times, memory_usages = [], [], []

    print("Loading dataset...")
    dataset = load_dataset("nanonets/key_information_extraction", split="test")
    train_ds, test_ds, val_ds = split_the_dataset(dataset)
    
    converted_train_dataset = [convert_to_conversation(sample) for sample in train_ds]
    converted_test_dataset = [convert_to_conversation(sample) for sample in test_ds]
    converted_val_dataset = [convert_to_conversation(sample) for sample in val_ds]

    model, tokenizer = configure_model(MODEL_NAME)
    
    # Training for the phase 1
    trainer_phase = configuration_for_training(model, tokenizer, converted_train_dataset, converted_val_dataset, 5)
    print("Phase 1 Finetuning Starting...")
    trainer_phase.train()
    used_memory_phase = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"Peak reserved memory Phase 1 = {used_memory_phase} GB.")

    print("Training completed. Saving and uploading the model...")
    save_finetuned_model(model, tokenizer, "finetuned_model")
    upload_to_huggingface_hub(model, tokenizer)
