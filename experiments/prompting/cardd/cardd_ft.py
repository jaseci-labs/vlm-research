#!/usr/bin/env python3
import argparse
import json
import torch
from shutil import make_archive
from datasets import load_from_disk, load_dataset
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import os
import sys

def main(args):
    # --- Load model ---
    print("🔄 Loading vision-language model:", args.model_name)
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=True,  # Use 4bit for memory efficiency
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False,
        finetune_language_layers   = True,
        finetune_attention_modules = False,
        finetune_mlp_modules       = True,
        r = 8,
        lora_alpha = 8,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    print("✅ Model loaded successfully.")

    print("🔄 Loading CarDD dataset...")
    train_dataset = load_dataset(args.datast_repo)['train']
    print("✅ Loaded:", len(train_dataset))

    def convert_to_conversation(sample):
        conversation = [
            {"role": "user",
             "content": [
                 {"type": "text", "text": args.prompt},
                 {"type": "image", "image": sample["image"]}]},
            {"role": "assistant",
             "content": [
                 {"type": "text", "text": sample["caption"]}]},
        ]
        return {"messages": conversation}

    converted_dataset = [convert_to_conversation(s) for s in train_dataset]

    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = converted_dataset,
        args = SFTConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 45,
            learning_rate = 5e-4,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = 1,
            optim = "paged_adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = args.save_dir,
            report_to = "none",
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 2048,
        ),
    )

    # --- Training ---
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB reserved before training.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used.")
    print(f"Peak reserved memory = {used_memory} GB ({used_percentage}%).")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB ({lora_percentage}%).")

    # --- Collect stats into dict ---
    stats = {
        "model_name": args.model_name,
        "save_dir": args.save_dir,
        "gpu": gpu_stats.name,
        "max_memory_gb": max_memory,
        "start_reserved_memory_gb": start_gpu_memory,
        "peak_reserved_memory_gb": used_memory,
        "peak_reserved_memory_training_gb": used_memory_for_lora,
        "peak_reserved_memory_percent": used_percentage,
        "peak_reserved_memory_training_percent": lora_percentage,
        "train_runtime_sec": trainer_stats.metrics["train_runtime"],
        "train_runtime_min": round(trainer_stats.metrics["train_runtime"] / 60, 2),
        "metrics": trainer_stats.metrics,  # includes loss, throughput, etc.
    }

    # --- Save JSON ---
    stats_file = os.path.join(args.save_dir, "training_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"📊 Training stats saved to {stats_file}")

    # --- Save model ---
    save_dir = args.save_dir
    print(f"💾 Saving model + tokenizer to {save_dir}")
    
    # After training
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    model.push_to_hub(args.repo_id, token=args.hf_token)
    tokenizer.push_to_hub(args.repo_id, token=args.hf_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a VLM with Unsloth")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base model to load (e.g., unsloth/Pixtral-12B-2409)")
    parser.add_argument("--datast_repo", type=str, default="RR32444/cardd_dataset",)
    parser.add_argument("--save_dir", type=str, default="flickr_px_finetune",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--prompt", type=str, default="Describe the image in detail.",
                        help="Prompt instruction for training")
    # parser.add_argument(
    #     "--exclude",
    #     nargs="+",
    #     type=int,
    #     help="List of indices to exclude from dataset"
    # )
    parser.add_argument("--hf_token", type=str, required=True,
                        help="Your Hugging Face access token")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Where to save the trained model")
    args = parser.parse_args()

    if not args.hf_token:
        print("❌ Error: Hugging Face token not provided.")
        sys.exit(1)

    if not args.repo_id:
        print("❌ Error: Hugging Face repo id not provided.")
        sys.exit(1)

    main(args)
