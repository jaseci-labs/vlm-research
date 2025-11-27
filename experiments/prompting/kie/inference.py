#!/usr/bin/env python3
"""
Run inference on fine-tuned KIE model.
Loads model from HuggingFace or local directory and generates predictions.
"""

import argparse
import json
import time
import threading
import torch
from typing import List, Dict, Any
from PIL import Image as PILImage
from datasets import load_from_disk
from unsloth import FastVisionModel
from transformers import TextIteratorStreamer
import re


def clean_json_output(raw_output: str) -> str:
    """
    Clean the model output to extract only the JSON object.
    Removes markdown code blocks, extra text, and formatting.
    
    Args:
        raw_output: Raw output from the model
        
    Returns:
        Clean JSON string
    """
    cleaned = re.sub(r'```json\s*', '', raw_output)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
        cleaned = cleaned[start_idx:end_idx + 1]
    
    cleaned = cleaned.strip()
    return cleaned


def run_inference(
    image,
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 256
):
    """
    Run inference on a single image with the given instruction.
    Uses greedy decoding for deterministic results.

    Args:
        image: PIL Image or image data
        model: Vision-language model
        tokenizer: Tokenizer
        instruction: Text prompt/instruction
        max_new_tokens: Maximum tokens to generate

    Returns:
        Tuple of (generated_caption, inference_time_s, peak_vram_mb)
    """
    try:
        # Build the chat prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
                ]
            }
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Prepare inputs and move to CUDA
        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        inputs.pop("token_type_ids", None)

        # Reset CUDA memory stats
        torch.cuda.reset_peak_memory_stats(device="cuda")

        # Start timing
        t_start = time.time()

        # Set up the streamer and launch generation in a thread
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = threading.Thread(
            target=model.generate,
            kwargs={
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
                "do_sample": False,  # Greedy decoding for deterministic results
                "top_p": 1.0,  # No nucleus truncation (reproducibility guardrail)
                "top_k": 0,  # No top-k filtering
            }
        )
        thread.start()

        # Collect tokens
        generated_caption = ""
        for token in streamer:
            generated_caption += token

        # Ensure generation is done
        thread.join()

        # Stop timing
        t_end = time.time()
        inference_time_s = t_end - t_start

        # Peak VRAM usage in bytes → convert to MiB
        peak_vram_bytes = torch.cuda.max_memory_allocated(device="cuda")
        peak_vram_mb = peak_vram_bytes / (1024 ** 2)

        # Clean the generated output to extract only JSON
        cleaned_caption = clean_json_output(generated_caption.strip())
        return cleaned_caption, inference_time_s, peak_vram_mb

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return "", 0.0, 0.0
def load_model(
    model_path: str,
    load_from_hf: bool = False,
    base_model: str = "unsloth/gemma-3-12b-it"
):
    """
    Load model and tokenizer.

    Args:
        model_path: Path to model (local directory or HF repo)
        load_from_hf: Whether to load base model and apply adapter
        base_model: Base model name (used if load_from_hf is True)

    Returns:
        Tuple of (model, tokenizer)
    """
    if load_from_hf:
        print(f"🔄 Loading base model: {base_model}")
        model, tokenizer = FastVisionModel.from_pretrained(
            base_model,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )

        print(f"🔄 Applying adapter from: {model_path}")
        model.load_adapter(model_path)
    else:
        print(f"🔄 Loading full model from: {model_path}")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )

    model.eval()
    print("✅ Model loaded successfully.")
    return model, tokenizer


def run_inference_batch(
    model,
    tokenizer,
    test_dataset,
    prompt: str,
    sample_indices: List[int],
    max_new_tokens: int = 256
) -> tuple[Dict[int, str], Dict[int, str], Dict[int, float], Dict[int, float]]:
    """
    Run inference on a batch of samples using greedy decoding.

    Args:
        model: Vision-language model
        tokenizer: Tokenizer
        test_dataset: Test dataset
        prompt: Instruction prompt
        sample_indices: List of sample indices to process
        max_new_tokens: Maximum tokens to generate

    Returns:
        Tuple of (predictions, ground_truths, inference_times, vram_usage)
    """
    predictions = {}
    ground_truths = {}
    inference_times = {}
    vram_usage = {}

    print(f"🚀 Running inference on {len(sample_indices)} samples...")
    print(f"   Using greedy decoding (deterministic)")

    for idx in sample_indices:
        print(f"\n📦 Processing sample {idx}...")
        sample = test_dataset[idx]

        # Decode bytes to PIL Image if needed
        from io import BytesIO
        if isinstance(sample['image'], bytes):
            image = PILImage.open(BytesIO(sample['image']))
        else:
            image = sample['image']

        # Run inference
        pred, inf_time, vram = run_inference(
            image,
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens
        )

        predictions[idx] = pred
        # KIE dataset has 'annotations' field containing dict with KIE fields
        annotations = sample.get('annotations', sample.get('ground_truth', sample.get('annotation', '')))
        # Convert dict to JSON string if needed
        if isinstance(annotations, dict):
            import json
            annotations = json.dumps(annotations)
        ground_truths[idx] = str(annotations)
        inference_times[idx] = inf_time
        vram_usage[idx] = vram

        print(f"   ⏱️  Inference time: {inf_time:.3f}s")
        print(f"   💾 VRAM usage: {vram:.2f} MB")
        print(f"   📝 Prediction: {pred[:100]}...")

    print("\n✅ Inference batch complete!")
    return predictions, ground_truths, inference_times, vram_usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on KIE model")

    # Model
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model (local or HF repo)")
    parser.add_argument("--load-from-hf", action="store_true",
                        help="Load base model and apply adapter from model-path")
    parser.add_argument("--base-model", type=str, default="unsloth/gemma-3-12b-it",
                        help="Base model (used with --load-from-hf)")

    # Data
    parser.add_argument("--test-dataset", type=str, default="./kie_splits/test",
                        help="Path to test dataset")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Instruction prompt for inference")
    parser.add_argument("--sample-indices", type=str, default="",
                        help="Comma-separated list of sample indices to process (empty = all samples)")

    # Inference parameters
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum tokens to generate")

    # Output
    parser.add_argument("--output-dir", type=str, default="./inference_results",
                        help="Directory to save inference results")

    args = parser.parse_args()

    # Parse sample indices - if empty, use all samples from test dataset
    if args.sample_indices.strip():
        sample_indices = [int(x.strip()) for x in args.sample_indices.split(",")]
    else:
        # Load test dataset to get all indices
        from datasets import load_from_disk
        test_ds = load_from_disk(args.test_dataset)
        sample_indices = list(range(len(test_ds)))
        print(f"📊 No sample indices specified - using all {len(sample_indices)} samples from test dataset")

    # Load model
    model, tokenizer = load_model(
        args.model_path,
        args.load_from_hf,
        args.base_model
    )

    # Load test dataset
    print(f"\n🔄 Loading test dataset from: {args.test_dataset}")
    test_dataset = load_from_disk(args.test_dataset)
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")

    # Run inference
    predictions, ground_truths, inference_times, vram_usage = run_inference_batch(
        model,
        tokenizer,
        test_dataset,
        args.prompt,
        sample_indices,
        args.max_new_tokens
    )

    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        "prompt": args.prompt,
        "model_path": args.model_path,
        "sample_indices": sample_indices,
        "predictions": predictions,
        "ground_truths": ground_truths,
        "inference_times": inference_times,
        "vram_usage": vram_usage,
    }

    output_file = os.path.join(args.output_dir, "inference_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n💾 Results saved to: {output_file}")
