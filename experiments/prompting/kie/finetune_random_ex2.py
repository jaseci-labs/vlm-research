#!/usr/bin/env python3
"""
Fine-tune Gemma-3-12B-IT vision-language model for KIE task.

NEW (for your research case):
- Train ONE model using MULTIPLE prompts from prompts.yaml/prompts.yml
- Prompt assignment mode: random quota (exact counts per prompt)
- Saves prompt_assignment.json so you can verify distribution (indexes per prompt)

Backward compatible:
- You can still pass --prompt "..." and it will behave like your old version.
"""

import argparse
import json
import os
import random
import torch
from datasets import load_from_disk
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


# -------------------------------
# Helper: Load prompts from YAML
# -------------------------------
def load_prompts_from_yaml(prompts_file: str):
    """
    Expected YAML format (like your previous extractor):
      key1:
        text: "prompt text..."
      key2:
        text: "prompt text..."

    Returns:
      prompt_keys: List[str]
      prompt_texts: List[str]
    """
    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            "PyYAML is required to read prompts file. Install with: pip install pyyaml"
        )

    with open(prompts_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Prompts file {prompts_file} is empty or invalid YAML dict.")

    prompt_keys = []
    prompt_texts = []
    for k, v in data.items():
        if isinstance(v, dict) and "text" in v and str(v["text"]).strip():
            prompt_keys.append(str(k))
            prompt_texts.append(str(v["text"]).strip())

    if len(prompt_keys) == 0:
        raise ValueError(f"No prompts found in {prompts_file}. Expected entries with a 'text' field.")

    return prompt_keys, prompt_texts


# ----------------------------------------------
# Helper: Create random quota prompt assignments
# ----------------------------------------------
def make_random_quota_assignment(n_samples: int, prompt_keys, seed: int):
    """
    Randomly assigns sample indices to prompts with exact quotas.

    Quotas rule:
      base = n_samples // n_prompts
      remainder = n_samples % n_prompts
      first 'remainder' prompts get base+1, others get base

    For n=591, k=7 => base=84 remainder=3 => 85,85,85,84,84,84,84 (in prompt order)

    Returns:
      quotas: dict(prompt_key -> quota)
      assignments: dict(prompt_key -> list of indices (0-based))
    """
    k = len(prompt_keys)
    base = n_samples // k
    remainder = n_samples % k

    quotas = {}
    for i, key in enumerate(prompt_keys):
        quotas[key] = base + (1 if i < remainder else 0)

    all_indices = list(range(n_samples))
    rng = random.Random(seed)
    rng.shuffle(all_indices)

    assignments = {}
    cursor = 0
    for key in prompt_keys:
        q = quotas[key]
        assignments[key] = sorted(all_indices[cursor:cursor + q])
        cursor += q

    # Sanity checks
    total = sum(len(v) for v in assignments.values())
    if total != n_samples:
        raise RuntimeError(f"Assignment bug: total assigned {total} != n_samples {n_samples}")
    for key in prompt_keys:
        if len(assignments[key]) != quotas[key]:
            raise RuntimeError(f"Assignment bug: {key} assigned {len(assignments[key])} != quota {quotas[key]}")

    return quotas, assignments


def make_random_unrestricted_assignment(n_samples: int, prompt_keys, seed: int):
    """
    Randomly assigns each sample to a prompt (independent draw).
    No quota restriction -> counts can be uneven.
    Returns:
      counts: dict(prompt_key -> count)
      assignments: dict(prompt_key -> list of indices (0-based))
    """
    rng = random.Random(seed)
    assignments = {k: [] for k in prompt_keys}

    for i in range(n_samples):
        k = rng.choice(prompt_keys)
        assignments[k].append(i)

    for k in assignments:
        assignments[k] = sorted(assignments[k])

    counts = {k: len(v) for k, v in assignments.items()}
    return counts, assignments



# Core training function (modified)

def finetune_model(
    model_name: str,
    train_dataset_path: str,
    eval_dataset_path: str,
    save_dir: str,

    # OLD single-prompt mode (backward compatible)
    prompt: str = None,

    # NEW multi-prompt mode
    prompts_file: str = None,
    prompt_assignment: str = "random_quota",  # currently supports: random_quota
    prompt_seed: int = 3407,
    save_prompt_assignment: bool = False,

    hf_token: str = None,
    repo_id: str = None,
    upload_to_hf: bool = False,

    # WandB config
    use_wandb: bool = False,
    wandb_entity: str = None,
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
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print(f"🔄 Loading vision-language model: {model_name}")
    print("=" * 80)

    # Initialize WandB if requested
    if use_wandb:
        try:
            import wandb
            wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                name=wandb_run_name,
                tags=wandb_tags or []
            )
            print(f"✅ WandB initialized: {wandb_entity}/{wandb_project}/{wandb_run_name}")
            report_to = "wandb"
        except ImportError:
            print("⚠️  WandB not installed. Install with: pip install wandb")
            report_to = "none"
        except Exception as e:
            print(f"⚠️  WandB initialization failed: {e}")
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
    n_train = len(train_dataset)
    print(f"✅ Training dataset loaded: {n_train} samples")

    print(f"\n🔄 Loading evaluation dataset from: {eval_dataset_path}")
    eval_dataset = load_from_disk(eval_dataset_path)
    n_eval = len(eval_dataset)
    print(f"✅ Evaluation dataset loaded: {n_eval} samples")

    
    # Decide prompts mode
    
    use_multi_prompts = prompts_file is not None and str(prompts_file).strip() != ""

    if use_multi_prompts:
        prompt_keys, prompt_texts = load_prompts_from_yaml(prompts_file)
        print(f"\n✅ Loaded {len(prompt_keys)} prompt(s) from: {prompts_file}")
        print("   Prompt keys:", ", ".join(prompt_keys))

        # if prompt_assignment != "random_quota":
        #     raise ValueError(f"Unsupported --prompt-assignment '{prompt_assignment}'. Use: random_quota")

        # quotas, assignments = make_random_quota_assignment(
        #     n_samples=n_train,
        #     prompt_keys=prompt_keys,
        #     seed=prompt_seed
        # )

        ################

        if prompt_assignment == "random_quota":
            quotas, assignments = make_random_quota_assignment(
                n_samples=n_train,
                prompt_keys=prompt_keys,
                seed=prompt_seed
            )

        elif prompt_assignment == "random_unrestricted":
            quotas, assignments = make_random_unrestricted_assignment(
                n_samples=n_train,
                prompt_keys=prompt_keys,
                seed=prompt_seed
            )

        else:
            raise ValueError(
                f"Unsupported --prompt-assignment '{prompt_assignment}'. Use: random_quota or random_unrestricted"
            )



        ##########

        # Build a per-index prompt lookup for fast access
        prompt_for_index = [None] * n_train
        key_to_text = dict(zip(prompt_keys, prompt_texts))
        for key, idxs in assignments.items():
            for i in idxs:
                prompt_for_index[i] = key_to_text[key]

        if any(p is None for p in prompt_for_index):
            raise RuntimeError("Prompt assignment failed: some training indices have no prompt assigned.")

        print("\n📌 Prompt quotas (train):")
        for k in prompt_keys:
            print(f"   - {k}: {quotas[k]}")

        # Save assignment file for verification (what your .sh prints)
        if save_prompt_assignment:
            assignment_out = os.path.join(save_dir, "prompt_assignment.json")
            payload = {
                "mode": prompt_assignment,
                "prompt_seed": prompt_seed,

                "train_size": n_train,
                "prompt_keys_in_order": prompt_keys,
                "train_prompt_counts": quotas,
                "train_assignments": assignments,

                "eval_size": n_eval,
                "eval_prompt_counts": eval_quotas,
                "eval_assignments": eval_assignments,
            }
            with open(assignment_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"\n✅ Saved prompt assignment to: {assignment_out}")

    else:
        # Old single prompt behavior
        if prompt is None or str(prompt).strip() == "":
            raise ValueError("You must provide either --prompt OR --prompts-file.")
        print(f"\n✅ Using single prompt mode (same prompt for all training samples).")

    
    # Convert samples -> conversation format
    
    def convert_to_conversation(sample, prompt_text: str):
        """Convert KIE sample to conversation format for Unsloth."""
        from PIL import Image
        from io import BytesIO

        # Decode bytes to PIL Image
        if isinstance(sample["image"], bytes):
            image = Image.open(BytesIO(sample["image"]))
        else:
            image = sample["image"]

        annotations = sample.get("annotations", sample.get("ground_truth", sample.get("annotation", "")))

        if isinstance(annotations, dict):
            annotations_text = json.dumps(annotations)
        else:
            annotations_text = str(annotations)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
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


    if use_multi_prompts:
        converted_train = [
            convert_to_conversation(train_dataset[i], prompt_for_index[i])
            for i in range(n_train)
        ]
    
        #   Eval: random_quota assignment too (equal counts, reproducible)
        if prompt_assignment == "random_quota":
            eval_quotas, eval_assignments = make_random_quota_assignment(
                n_samples=n_eval,
                prompt_keys=prompt_keys,
                seed=prompt_seed + 1  # use different seed stream than train
            )
        else:
            eval_quotas, eval_assignments = make_random_unrestricted_assignment(
                n_samples=n_eval,
                prompt_keys=prompt_keys,
                seed=prompt_seed + 1
            )
    
        # Build per-index eval prompt lookup
        eval_prompt_for_index = [None] * n_eval
        key_to_text = dict(zip(prompt_keys, prompt_texts))
        for key, idxs in eval_assignments.items():
            for i in idxs:
                eval_prompt_for_index[i] = key_to_text[key]
    
        if any(p is None for p in eval_prompt_for_index):
            raise RuntimeError("Eval prompt assignment failed: some eval indices have no prompt assigned.")
    
        converted_eval = [
            convert_to_conversation(eval_dataset[i], eval_prompt_for_index[i])
            for i in range(n_eval)
        ]
    
        print("✅ Train converted with mixed prompts (random_quota).")
        print("✅ Eval converted with mixed prompts (random_quota).")
        print("📌 Eval prompt quotas:")
        for k in prompt_keys:
            print(f"   - {k}: {eval_quotas[k]}")

    else:
        converted_train = [convert_to_conversation(s, prompt) for s in train_dataset]
        converted_eval = [convert_to_conversation(s, prompt) for s in eval_dataset]
        print("✅ Train/Eval converted with single prompt.")



    FastVisionModel.for_training(model)

    warmup_steps = int(max_steps * warmup_ratio)
    eval_steps = max(10, max_steps // 10)
    save_steps = eval_steps

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
            report_to=report_to,
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

    print("\n" + "=" * 80)
    print("🚀 Starting fine-tuning...")
    print("=" * 80)

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU: {gpu_stats.name}")
    print(f"Max memory: {max_memory} GB")
    print(f"Reserved memory before training: {start_gpu_memory} GB\n")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print("\n" + "=" * 80)
    print("✅ Fine-tuning complete!")
    print("=" * 80)
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds "
          f"({trainer_stats.metrics['train_runtime']/60:.2f} minutes)")
    print(f"Peak reserved memory: {used_memory} GB ({used_percentage}%)")
    print(f"Peak reserved memory for training: {used_memory_for_lora} GB ({lora_percentage}%)")

    # Stats
    stats = {
        "model_name": model_name,
        "save_dir": save_dir,
        "mode": "multi_prompts" if use_multi_prompts else "single_prompt",
        "single_prompt": prompt if not use_multi_prompts else None,
        "prompts_file": prompts_file if use_multi_prompts else None,
        "prompt_assignment": prompt_assignment if use_multi_prompts else None,
        "prompt_seed": prompt_seed if use_multi_prompts else None,
        "lora_config": {"r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout},
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
            "seed": seed,
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
        "train_samples": n_train,
        "eval_samples": n_eval,
    }

    stats_file = os.path.join(save_dir, "training_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    print(f"\n📊 Training stats saved to: {stats_file}")

    print(f"\n💾 Saving model and tokenizer to: {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("✅ Model saved successfully.")

    if upload_to_hf and hf_token and repo_id:
        print(f"\n🔄 Uploading model to HuggingFace: {repo_id}")
        try:
            model.push_to_hub(repo_id, token=hf_token)
            tokenizer.push_to_hub(repo_id, token=hf_token)
            print(f"✅ Model uploaded successfully to: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"❌ Failed to upload model to HuggingFace: {e}")

    print("\n" + "=" * 80)
    print("🎉 Fine-tuning process completed!")
    print("=" * 80)

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

    # Old single prompt (still supported)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single instruction prompt (old behavior).")

    # New multi prompts
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="Path to prompts.yaml/prompts.yml containing multiple prompts.")
    parser.add_argument("--prompt-assignment", type=str, default="random_quota",
                        help="Prompt assignment mode. Use: random_quota or random_unrestricted")
    parser.add_argument("--prompt-seed", type=int, default=3407,
                        help="Seed for prompt assignment randomness (reproducibility).")
    parser.add_argument("--save-prompt-assignment", type=str, default="false",
                        help="If true, saves prompt_assignment.json into save-dir.")

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
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="WandB entity/team name")
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

    save_prompt_assignment_bool = str(args.save_prompt_assignment).strip().lower() in ("1", "true", "yes", "y")

    finetune_model(
        model_name=args.model_name,
        train_dataset_path=args.train_dataset,
        eval_dataset_path=args.eval_dataset,
        save_dir=args.save_dir,
        prompt=args.prompt,

        prompts_file=args.prompts_file,
        prompt_assignment=args.prompt_assignment,
        prompt_seed=args.prompt_seed,
        save_prompt_assignment=save_prompt_assignment_bool,

        hf_token=args.hf_token,
        repo_id=args.repo_id,
        upload_to_hf=args.upload_to_hf,

        use_wandb=args.use_wandb,
        wandb_entity=args.wandb_entity,
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
