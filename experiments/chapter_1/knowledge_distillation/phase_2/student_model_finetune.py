from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
import torch

from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile
import json
import os

from huggingface_hub import login
import wandb

ImageFile.LOAD_TRUNCATED_IMAGES = True
wandb.init(
    project="Gemma 3 4B Distillation Phase 2",
    name="Gemma 3 4B Distillation Phase 2",
    group="gemma tests",
    tags=["gemma","vision", "finetune"],
    notes="Testing gemma 3 4B Unsloth",
    config={
        "model": "Gemma 3 4B",
    },
)

MODEL_NAME = "unsloth/gemma-3-4b-it"
JSON_FILE_PATH = "res.json"
IMAGES_PATH = "image_dataset"
instruction = "Descibe the damages of the car.",

def dataset_split(json_path, test_size=0.2, random_state=42):
    with open(json_path, "r") as f:
        data = json.load(f)

    data_list = []
    for image_path, info in data.items():
        entry = {
            "image": image_path,
            **info
        }
        data_list.append(entry)

    train_data, test_val_data = train_test_split(data_list, test_size=test_size, random_state=random_state)
    val_data, test_data = train_test_split(test_val_data, test_size=0.5, random_state=random_state)

    return train_data, val_data, test_data

def convert_to_conversation(phase, sample):
    if phase == 1:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(sample["caption"]["predictions"], indent=2)},
                        {"type": "image", "image": sample["image"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": json.dumps(sample["caption"]["report"], indent=2)}],
                },
            ]
        }
    elif phase == 2:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image", "image": sample["image"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": json.dumps(sample["caption"], indent=2)}],
                },
            ]
        }


# should be edited
def get_custom_dataset(data, phase):
    custom_dataset = []
    # here data is list of key value pairs
    for sample in data:
        full_path = sample["image"]
        if os.path.exists(full_path):
            try:
                image = Image.open(full_path)
                custom_dataset.append(
                    convert_to_conversation(
                        phase=phase, 
                        sample={"image": image, "caption": {
                                "predictions": sample["predictions"],
                                "report": sample["report"]
                            }
                        }
                    )
                )
            except Exception as e:
                print(f"[ERROR] Could not load image {full_path}: {e}")
        else:
            print(f"[WARNING] Image not found: {full_path}")
    return custom_dataset


def configure_model(MODEL_NAME):
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit = False,
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


def configuration_for_training(model, tokenizer, train_data, val_data, steps=30):
    FastVisionModel.for_training(model)

    callbacks = [
    EarlyStoppingCallback(early_stopping_patience=5),
    ]
    
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
            max_steps = steps,
            # num_train_epochs = 1, # Set this instead of max_steps for full training runs
            learning_rate = 2e-4,
            fp16 = False,
            bf16 = True,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "wandb",
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 2048,
            eval_strategy="steps",
            eval_steps=2,
    ),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    callbacks=callbacks,
    )
    return trainer

def save_finetuned_model(model, tokenizer, model_name):
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)


def upload_to_huggingface_hub(model, processor):
    model.push_to_hub("Malitha/Gemma3-car-damage-model-4B")
    processor.push_to_hub("Malitha/Gemma3-car-damage-model-4B")


# Main execution
if __name__ == "__main__":
    login(token="hf_XXXXXXXXXXXXXXXXXXXXXXX")  # Replace with your Hugging Face token

    print("Loading dataset...")
    train_data, val_data, test_data = dataset_split(JSON_FILE_PATH, test_size=0.2, random_state=42)
    with open("test_data.json", "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"Train data size: {len(train_data)}, Validation data size: {len(val_data)}, Test data size: {len(test_data)}")

    # Splitting the dataset into two phases
    train_data_phase_1 = get_custom_dataset(train_data, phase=1)
    train_data_phase_2 = get_custom_dataset(train_data, phase=2)
    val_data_phase_1 = get_custom_dataset(val_data, phase=1)
    val_data_phase_2 = get_custom_dataset(val_data, phase=2)
    test_data = get_custom_dataset(test_data, phase=2)

    model, tokenizer = configure_model(MODEL_NAME)
    
    # Training for the phase 1
    trainer_phase_1 = configuration_for_training(model, tokenizer, train_data_phase_1, val_data_phase_1, 40)
    print("Phase 1 Finetuning Starting...")
    trainer_phase_1.train()
    used_memory_phase_1 = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"Peak reserved memory Phase 1 = {used_memory_phase_1} GB.")

    # Training for the phase 2
    trainer_phase_2 = configuration_for_training(model, tokenizer, train_data_phase_2, val_data_phase_2, 200)
    print("Phase 2 Finetuning Starting...")
    trainer_phase_2.train()
    used_memory_phase_2 = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"Peak reserved memory Phase 2 = {used_memory_phase_2} GB.")

    print("Training completed. Saving and uploading the model...")
    save_finetuned_model(model, tokenizer, "finetuned_model")
    upload_to_huggingface_hub(model, tokenizer)