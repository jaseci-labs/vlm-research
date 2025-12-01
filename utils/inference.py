#!/usr/bin/env python3
"""
Shared inference utilities for VLM experiments.
"""

import time
import threading
from typing import Tuple, Dict, List, Any
from PIL import Image
from io import BytesIO
import torch

from .formatting import clean_json_output


def run_inference(
    image: Image.Image,
    model: Any,
    tokenizer: Any,
    instruction: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    top_p: float = 1.0,
    top_k: int = 0
) -> Tuple[str, float, float]:
    """
    Run inference on a single image with the given instruction.
    Uses greedy decoding by default for deterministic results.

    Args:
        image: PIL Image object
        model: Vision-language model (Unsloth FastVisionModel)
        tokenizer: Tokenizer for the model
        instruction: Text prompt/instruction
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to use sampling (False = greedy decoding)
        top_p: Nucleus sampling parameter (only if do_sample=True)
        top_k: Top-k sampling parameter (only if do_sample=True)

    Returns:
        Tuple of (generated_text, inference_time_seconds, peak_vram_mb)

    Example:
        >>> text, time_s, vram_mb = run_inference(image, model, tokenizer, prompt)
        >>> print(f"Generated: {text}, Time: {time_s:.2f}s, VRAM: {vram_mb:.0f}MB")
    """
    from transformers import TextIteratorStreamer

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
                "do_sample": do_sample,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
        thread.start()

        # Collect tokens
        generated_text = ""
        for token in streamer:
            generated_text += token

        # Ensure generation is done
        thread.join()

        # Stop timing
        t_end = time.time()
        inference_time_s = t_end - t_start

        # Peak VRAM usage in bytes → convert to MiB
        peak_vram_bytes = torch.cuda.max_memory_allocated(device="cuda")
        peak_vram_mb = peak_vram_bytes / (1024 ** 2)

        # Clean the generated output to extract only JSON
        cleaned_text = clean_json_output(generated_text.strip())
        return cleaned_text, inference_time_s, peak_vram_mb

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return "", 0.0, 0.0


def run_inference_batch(
    model: Any,
    tokenizer: Any,
    test_dataset: Any,
    prompt: str,
    sample_indices: List[int],
    max_new_tokens: int = 256
) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, float], Dict[int, float]]:
    """
    Run inference on a batch of samples using greedy decoding.

    Args:
        model: Vision-language model
        tokenizer: Tokenizer
        test_dataset: Test dataset with 'image' and 'annotations' fields
        prompt: Instruction prompt
        sample_indices: List of sample indices to process
        max_new_tokens: Maximum tokens to generate

    Returns:
        Tuple of (predictions, ground_truths, inference_times, vram_usage)
        Each is a dict mapping sample index to value.

    Example:
        >>> preds, gts, times, vram = run_inference_batch(
        ...     model, tokenizer, test_ds, prompt, [0, 1, 2, 3, 4]
        ... )
    """
    import json

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
        if isinstance(sample['image'], bytes):
            image = Image.open(BytesIO(sample['image']))
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

        # Get ground truth - handle different field names
        annotations = sample.get(
            'annotations',
            sample.get('ground_truth', sample.get('annotation', ''))
        )

        # Convert dict to JSON string if needed
        if isinstance(annotations, dict):
            annotations = json.dumps(annotations)
        ground_truths[idx] = str(annotations)

        inference_times[idx] = inf_time
        vram_usage[idx] = vram

        print(f"   ⏱️  Inference time: {inf_time:.3f}s")
        print(f"   💾 VRAM usage: {vram:.2f} MB")
        print(f"   📝 Prediction: {pred[:100]}...")

    print("\n✅ Inference batch complete!")
    return predictions, ground_truths, inference_times, vram_usage


def get_gpu_stats() -> Dict[str, float]:
    """
    Get current GPU memory statistics.

    Returns:
        Dict with GPU stats (name, max_memory_gb, reserved_gb, allocated_gb)
    """
    if not torch.cuda.is_available():
        return {"available": False}

    gpu_props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": gpu_props.name,
        "max_memory_gb": round(gpu_props.total_memory / 1024 / 1024 / 1024, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024 / 1024 / 1024, 3),
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3),
    }
