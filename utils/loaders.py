#!/usr/bin/env python3
"""
Shared model and dataset loading utilities for VLM research experiments.
"""

import os
from typing import Dict, Optional, Tuple, Any
from PIL import Image
from io import BytesIO


def load_model(
    model_path: str,
    load_from_hf: bool = False,
    base_model: str = "unsloth/gemma-3-12b-it",
    load_in_4bit: bool = True,
    use_gradient_checkpointing: str = "unsloth"
) -> Tuple[Any, Any]:
    """
    Load a vision-language model and tokenizer using Unsloth.

    Args:
        model_path: Path to model (local directory or HuggingFace repo for adapter)
        load_from_hf: Whether to load base model and apply adapter from model_path
        base_model: Base model name (used if load_from_hf is True)
        load_in_4bit: Use 4-bit quantization for memory efficiency
        use_gradient_checkpointing: Gradient checkpointing mode ("unsloth" recommended)

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load_model("./kie_finetuned/baseline")
        >>> model, tokenizer = load_model("Gayanukaa/vlm-finetunes-baseline",
        ...                                load_from_hf=True)
    """
    from unsloth import FastVisionModel

    if load_from_hf:
        print(f"🔄 Loading base model: {base_model}")
        model, tokenizer = FastVisionModel.from_pretrained(
            base_model,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        print(f"🔄 Applying adapter from: {model_path}")
        model.load_adapter(model_path)
    else:
        print(f"🔄 Loading full model from: {model_path}")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    model.eval()
    print("✅ Model loaded successfully.")
    return model, tokenizer


def load_base_model_for_training(
    model_name: str = "unsloth/gemma-3-12b-it",
    load_in_4bit: bool = True,
    use_gradient_checkpointing: str = "unsloth",
    # LoRA config
    lora_r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.01,
    finetune_vision_layers: bool = False,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = False,
    finetune_mlp_modules: bool = True,
    seed: int = 3407
) -> Tuple[Any, Any]:
    """
    Load a base model and configure it for LoRA fine-tuning.

    Args:
        model_name: Base model identifier (e.g., unsloth/gemma-3-12b-it)
        load_in_4bit: Use 4-bit quantization
        use_gradient_checkpointing: Gradient checkpointing mode
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
        finetune_vision_layers: Whether to fine-tune vision encoder
        finetune_language_layers: Whether to fine-tune language layers
        finetune_attention_modules: Whether to fine-tune attention
        finetune_mlp_modules: Whether to fine-tune MLP layers
        seed: Random seed for reproducibility

    Returns:
        Tuple of (model, tokenizer) configured for training
    """
    from unsloth import FastVisionModel

    print("=" * 80)
    print(f"🔄 Loading vision-language model: {model_name}")
    print("=" * 80)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=finetune_vision_layers,
        finetune_language_layers=finetune_language_layers,
        finetune_attention_modules=finetune_attention_modules,
        finetune_mlp_modules=finetune_mlp_modules,
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

    return model, tokenizer


def load_and_split_dataset(
    dataset_name: str,
    output_dir: str,
    train_ratio: float = 0.6,
    eval_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    hf_split: str = "test"
) -> Dict[str, Any]:
    """
    Load a HuggingFace dataset and split it into train/eval/test sets.
    Saves splits to disk for reuse across experiments.

    Args:
        dataset_name: HuggingFace dataset identifier
        output_dir: Directory to save the split datasets
        train_ratio: Proportion for training (default 0.6)
        eval_ratio: Proportion for evaluation during training (default 0.2)
        test_ratio: Proportion for testing (default 0.2)
        seed: Random seed for reproducibility
        hf_split: Which split to load from HuggingFace (default "test")

    Returns:
        Dictionary with 'train', 'eval', 'test' datasets

    Example:
        >>> splits = load_and_split_dataset(
        ...     "nanonets/key_information_extraction",
        ...     "./kie_splits"
        ... )
        >>> train_ds = splits["train"]
    """
    from datasets import load_dataset

    assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + eval_ratio + test_ratio must equal 1.0"

    print(f"🔄 Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=hf_split)
    print(f"✅ Loaded {len(dataset)} samples")

    # First split: separate out test set
    split1 = dataset.train_test_split(test_size=test_ratio, seed=seed)
    test_dataset = split1['test']
    train_eval_dataset = split1['train']

    # Second split: separate train and eval from remaining data
    eval_from_remaining = eval_ratio / (train_ratio + eval_ratio)
    split2 = train_eval_dataset.train_test_split(test_size=eval_from_remaining, seed=seed)
    train_dataset = split2['train']
    eval_dataset = split2['test']

    print(f"✅ Split complete:")
    print(f"   - Training set: {len(train_dataset)} samples ({train_ratio*100:.1f}%)")
    print(f"   - Evaluation set: {len(eval_dataset)} samples ({eval_ratio*100:.1f}%)")
    print(f"   - Test set: {len(test_dataset)} samples ({test_ratio*100:.1f}%)")

    # Save splits to disk
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train")
    eval_path = os.path.join(output_dir, "eval")
    test_path = os.path.join(output_dir, "test")

    train_dataset.save_to_disk(train_path)
    eval_dataset.save_to_disk(eval_path)
    test_dataset.save_to_disk(test_path)

    print(f"💾 Datasets saved to: {output_dir}")

    return {
        "train": train_dataset,
        "eval": eval_dataset,
        "test": test_dataset
    }


def load_dataset_from_disk(dataset_path: str) -> Any:
    """
    Load a dataset from disk (previously saved with save_to_disk).

    Args:
        dataset_path: Path to the saved dataset directory

    Returns:
        HuggingFace Dataset object
    """
    from datasets import load_from_disk

    print(f"🔄 Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    return dataset


def decode_image(image_data) -> Image.Image:
    """
    Decode various image formats to PIL Image.

    Args:
        image_data: bytes, numpy array, or PIL Image

    Returns:
        PIL Image object
    """
    if isinstance(image_data, bytes):
        return Image.open(BytesIO(image_data))
    elif isinstance(image_data, Image.Image):
        return image_data
    else:
        # Assume numpy array
        return Image.fromarray(image_data)
