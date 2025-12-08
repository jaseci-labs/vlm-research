#!/usr/bin/env python3
"""
Load and split the Flickr30k dataset into train (60%), eval (20%), and test (20%) sets.
Saves splits to disk for reuse across finetuning, evaluation, and inference.
"""

import os
from datasets import load_dataset

def load_and_split_dataset(
    dataset_name: str = "nlphuji/flickr30k",
    output_dir: str = "./flickr30k_splits",
    train_ratio: float = 0.6,
    eval_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Load the Flickr30k dataset and split it into train, eval, and test sets.

    Args:
        dataset_name: Hugging Face dataset name
        output_dir: Directory to save the split datasets
        train_ratio: Proportion for training (default 0.6)
        eval_ratio: Proportion for evaluation during training (default 0.2)
        test_ratio: Proportion for testing (default 0.2)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with train, eval, and test datasets
    """
    assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + eval_ratio + test_ratio must equal 1.0"

    print(f"🔄 Loading dataset: {dataset_name}")
    # For Flickr30k we load a single split and then re-split it ourselves.
    # Adjust "train" to the correct source split if your HF dataset uses a different name.
    dataset = load_dataset(dataset_name, split="test")
    print(f"✅ Loaded {len(dataset)} samples")

    # First split: separate out test set
    train_eval_test_ratio = test_ratio
    split1 = dataset.train_test_split(test_size=train_eval_test_ratio, seed=seed)
    test_dataset = split1["test"]
    train_eval_dataset = split1["train"]

    # Second split: separate train and eval from the remaining data
    # eval should be eval_ratio / (train_ratio + eval_ratio) of the train_eval_dataset
    eval_from_remaining = eval_ratio / (train_ratio + eval_ratio)
    split2 = train_eval_dataset.train_test_split(
        test_size=eval_from_remaining,
        seed=seed,
    )
    train_dataset = split2["train"]
    eval_dataset = split2["test"]

    print("✅ Split complete:")
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
    print(f"   - Train: {train_path}")
    print(f"   - Eval: {eval_path}")
    print(f"   - Test: {test_path}")

    return {
        "train": train_dataset,
        "eval": eval_dataset,
        "test": test_dataset,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and split Flickr30k dataset")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="flickr30k",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./flickr30k_splits",
        help="Directory to save split datasets",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Training set ratio (default: 0.6)",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.2,
        help="Evaluation set ratio (default: 0.2)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test set ratio (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    load_and_split_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
