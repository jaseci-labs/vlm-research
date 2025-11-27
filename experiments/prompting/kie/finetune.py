#!/usr/bin/env python3
"""
Fine-tune Gemma-3-12B-IT vision-language model for KIE task.
Supports multiple prompts from prompts.yml and optional HuggingFace upload.
"""

import argparse
import json
import os
import torch
from datasets import load_from_disk
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


def finetune_model(
    model_name: str,
    train_dataset_path: str,
    eval_dataset_path: str,
    save_dir: str,
    prompt: str,
    hf_token: str = None,
    repo_id: str = None,
    upload_to_hf: bool = False,
    # WandB config
    use_wandb: bool = False,
    wandb_project: str = "kie-finetuning",
    wandb_run_name: str = None,
    wandb_tags: list = None,
    # LoRA config
    lora_r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.01,
    # Training hyperparameters
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.1,
    max_steps: int = 100,
    fp16: bool = True,
    optim: str = "adamw_8bit",
    lr_scheduler_type: str = "cosine",
    weight_decay: float = 0.01,
    max_seq_length: int = 2048,
    seed: int = 3407
):
    """
    Fine-tune a vision-language model for KIE.

    Args:
        model_name: Base model identifier (e.g., unsloth/gemma-3-12b-it)
        train_dataset_path: Path to training dataset
        eval_dataset_path: Path to evaluation dataset
        save_dir: Directory to save fine-tuned model
        prompt: Instruction prompt for the model
        hf_token: HuggingFace token for uploading
        repo_id: HuggingFace repository ID for uploading
        upload_to_hf: Whether to upload model to HuggingFace
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        learning_rate: Learning rate for optimizer
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        warmup_ratio: Warmup ratio for learning rate scheduler
        max_steps: Maximum training steps
        fp16: Use FP16 precision
        optim: Optimizer type
        lr_scheduler_type: Learning rate scheduler type
        weight_decay: Weight decay for optimizer
        max_seq_length: Maximum sequence length
        seed: Random seed
    """

    print("="*80)
    print(f"🔄 Loading vision-language model: {model_name}")
    print("="*80)

    # Initialize WandB if requested
    if use_wandb:
        try:
            import wandb
            
            # Initialize WandB - minimal config for loss tracking only
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                tags=wandb_tags or []
            )
            print(f"✅ WandB initialized: {wandb_project}/{wandb_run_name}")
            report_to = "wandb"
        except ImportError:
            print("⚠️  WandB not installed. Install with: pip install wandb")
            report_to = "none"
        except Exception as e:
            print(f"⚠️  WandB initialization failed: {e}")
            print("   Possible solutions:")
            print("   1. Run: wandb login")
            print("   2. Set: export WANDB_API_KEY=your_key")
            print("   3. Set USE_WANDB=false in the script")
            print("   Continuing without WandB logging...")
            report_to = "none"
    else:
        report_to = "none"

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=False,
        finetune_mlp_modules=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )

    print("✅ Model loaded successfully.")
    print(f"   - LoRA Config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

    print(f"\n🔄 Loading training dataset from: {train_dataset_path}")
    train_dataset = load_from_disk(train_dataset_path)
    print(f"✅ Training dataset loaded: {len(train_dataset)} samples")

    print(f"\n🔄 Loading evaluation dataset from: {eval_dataset_path}")
    eval_dataset = load_from_disk(eval_dataset_path)
    print(f"✅ Evaluation dataset loaded: {len(eval_dataset)} samples")

    def convert_to_conversation(sample):
        """Convert KIE sample to conversation format for Unsloth.

        The KIE dataset has:
        - 'image': binary/bytes that needs to be decoded to PIL Image
        - 'annotations': dict with KIE fields (date, seller_name, total_amount, etc.)
        """
        from PIL import Image
        from io import BytesIO
        import json

        # Decode bytes to PIL Image
        if isinstance(sample["image"], bytes):
            image = Image.open(BytesIO(sample["image"]))
        else:
            image = sample["image"]

        # Get annotations (use 'annotations' field, fallback to 'ground_truth' or 'annotation')
        annotations = sample.get("annotations", sample.get("ground_truth", sample.get("annotation", "")))

        # Convert dict to JSON string if needed
        if isinstance(annotations, dict):
            annotations_text = json.dumps(annotations)
        else:
            annotations_text = str(annotations)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": annotations_text}
                ]
            },
        ]
        return {"messages": conversation}

    print("\n🔄 Converting datasets to conversation format...")
    converted_train = [convert_to_conversation(s) for s in train_dataset]
    converted_eval = [convert_to_conversation(s) for s in eval_dataset]
    print("✅ Datasets converted successfully.")

    FastVisionModel.for_training(model)

    # Calculate warmup steps
    warmup_steps = int(max_steps * warmup_ratio)
    
    # Calculate eval and save steps to ensure compatibility with load_best_model_at_end
    eval_steps = max(10, max_steps // 10)
    save_steps = eval_steps  # Make save_steps equal to eval_steps to ensure they're compatible

    print(f"\n🔄 Setting up trainer...")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Batch size: {per_device_train_batch_size}")
    print(f"   - Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   - Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    print(f"   - Max steps: {max_steps}")
    print(f"   - Warmup steps: {warmup_steps}")
    print(f"   - Eval steps: {eval_steps}")
    print(f"   - Save steps: {save_steps}")
    print(f"   - Optimizer: {optim}")
    print(f"   - LR scheduler: {lr_scheduler_type}")
    print(f"   - Weight decay: {weight_decay}")
    print(f"   - Precision: {'FP16' if fp16 else 'BF16'}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_train,
        eval_dataset=converted_eval,
        args=SFTConfig(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=fp16 and not is_bf16_supported(),
            bf16=is_bf16_supported() and not fp16,
            logging_steps=5,
            optim=optim,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            seed=seed,
            output_dir=save_dir,
            report_to=report_to,  # Use WandB or none
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=max_seq_length,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
        ),
    )

    print("\n" + "="*80)
    print("🚀 Starting fine-tuning...")
    print("="*80)

    # Get GPU stats before training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU: {gpu_stats.name}")
    print(f"Max memory: {max_memory} GB")
    print(f"Reserved memory before training: {start_gpu_memory} GB")
    print()

    trainer_stats = trainer.train()

    # Get GPU stats after training
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print("\n" + "="*80)
    print("✅ Fine-tuning complete!")
    print("="*80)
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds "
          f"({trainer_stats.metrics['train_runtime']/60:.2f} minutes)")
    print(f"Peak reserved memory: {used_memory} GB ({used_percentage}%)")
    print(f"Peak reserved memory for training: {used_memory_for_lora} GB ({lora_percentage}%)")

    # Collect stats
    stats = {
        "model_name": model_name,
        "save_dir": save_dir,
        "prompt": prompt,
        "lora_config": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout
        },
        "training_config": {
            "learning_rate": learning_rate,
            "batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": per_device_train_batch_size * gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "optimizer": optim,
            "lr_scheduler": lr_scheduler_type,
            "weight_decay": weight_decay,
            "fp16": fp16,
            "seed": seed
        },
        "gpu_stats": {
            "name": gpu_stats.name,
            "max_memory_gb": max_memory,
            "start_reserved_memory_gb": start_gpu_memory,
            "peak_reserved_memory_gb": used_memory,
            "peak_reserved_memory_training_gb": used_memory_for_lora,
            "peak_reserved_memory_percent": used_percentage,
            "peak_reserved_memory_training_percent": lora_percentage,
        },
        "training_metrics": trainer_stats.metrics,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
    }

    # Save stats
    stats_file = os.path.join(save_dir, "training_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"\n📊 Training stats saved to: {stats_file}")

    # Save model
    print(f"\n💾 Saving model and tokenizer to: {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("✅ Model saved successfully.")

    # Upload to HuggingFace if requested
    if upload_to_hf and hf_token and repo_id:
        print(f"\n🔄 Uploading model to HuggingFace: {repo_id}")
        try:
            model.push_to_hub(repo_id, token=hf_token)
            tokenizer.push_to_hub(repo_id, token=hf_token)
            print(f"✅ Model uploaded successfully to: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"❌ Failed to upload model to HuggingFace: {e}")

    print("\n" + "="*80)
    print("🎉 Fine-tuning process completed!")
    print("="*80)

    # Finish WandB run
    if use_wandb and "wandb" in report_to:
        try:
            import wandb
            wandb.finish()
            print("✅ WandB run completed and logged")
        except Exception as e:
            print(f"⚠️  Warning: Failed to finish WandB run: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune VLM for KIE")

    # Model and data
    parser.add_argument("--model-name", type=str, default="unsloth/gemma-3-12b-it",
                        help="Base model to fine-tune")
    parser.add_argument("--train-dataset", type=str, default="./kie_splits/train",
                        help="Path to training dataset")
    parser.add_argument("--eval-dataset", type=str, default="./kie_splits/eval",
                        help="Path to evaluation dataset")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory to save fine-tuned model")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Instruction prompt for training")

    # HuggingFace upload
    parser.add_argument("--upload-to-hf", action="store_true",
                        help="Upload model to HuggingFace Hub")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace access token")
    parser.add_argument("--repo-id", type=str, default=None,
                        help="HuggingFace repository ID")

    # WandB logging
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable WandB logging for loss tracking")
    parser.add_argument("--wandb-project", type=str, default="kie-finetuning",
                        help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="WandB run name")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=[],
                        help="WandB tags for the run")

    # LoRA config
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=8,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.01,
                        help="LoRA dropout")

    # Training hyperparameters
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device training batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum training steps")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 precision")
    parser.add_argument("--optim", type=str, default="adamw_8bit",
                        help="Optimizer")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        help="Learning rate scheduler")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=3407,
                        help="Random seed")

    args = parser.parse_args()

    finetune_model(
        model_name=args.model_name,
        train_dataset_path=args.train_dataset,
        eval_dataset_path=args.eval_dataset,
        save_dir=args.save_dir,
        prompt=args.prompt,
        hf_token=args.hf_token,
        repo_id=args.repo_id,
        upload_to_hf=args.upload_to_hf,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        fp16=args.fp16,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler,
        weight_decay=args.weight_decay,
        max_seq_length=args.max_seq_length,
        seed=args.seed
    )
