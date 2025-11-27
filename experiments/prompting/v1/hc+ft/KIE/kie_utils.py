from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from transformers import TextIteratorStreamer
import threading
from sentence_transformers import SentenceTransformer, util
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice   import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import pandas as pd
import time
import torch
from __future__ import annotations
from typing import List, Union
from Levenshtein import distance as edit_distance
from tqdm import tqdm
import json
import pandas as pd

# ----- Define helper data structures -----

class Field:
    def __init__(self, label: str, value: str):
        self.label = label
        self.value = value

class GroundTruth:
    def __init__(self, fields: List[Field]):
        self.fields = fields

class Prediction:
    def __init__(self, fields: List[Field], gt: GroundTruth):
        self.fields = fields
        self.gt = gt

    def _get_pred_field_by_label(self, label: str):
        for field in self.fields:
            if field.label == label:
                return field
        return None

# ----- Scoring function -----

def get_kie_metrics(predictions: List[Prediction]) -> tuple[float, List[float]]:
    """
    Compute Levenshtein similarity per prediction (sample-level),
    averaged across fields, and return:
        - overall average score
        - list of per-sample scores
    """
    sample_scores = []

    for pred in tqdm(predictions, desc="Computing KIE metrics", leave=False):
        gt_fields = pred.gt.fields
        field_scores = []

        for gt_field in gt_fields:
            pred_field = pred._get_pred_field_by_label(gt_field.label)
            if pred_field is None or pred_field == "":
                pred_value = ""
            else:
                pred_value = pred_field.value

            pred_value = str(pred_value)
            gt_value = str(gt_field.value)

            dist = edit_distance(pred_value, gt_value)
            max_len = max(len(pred_value), len(gt_value))
            if max_len == 0:
                field_scores.append(1.0)
            else:
                field_scores.append(1 - (dist / max_len))

        sample_avg = sum(field_scores) / len(field_scores) if field_scores else 0.0
        sample_scores.append(sample_avg)

    overall_avg = sum(sample_scores) / len(sample_scores) if sample_scores else 0.0
    return overall_avg, sample_scores

# ----- Evaluation wrapper -----

def evaluate_kie_predictions(preds: List[Union[str, dict]], gts: List[dict]) -> tuple[float, List[float]]:
    """
    Evaluate KIE predictions against ground truth using Levenshtein similarity.

    Returns:
        - Average accuracy score across all samples.
        - List of individual sample-level accuracy scores.
    """
    assert len(preds) == len(gts), "Predictions and ground truth lists must be the same length."

    prediction_objects = []

    for pred_raw, gt_json in zip(preds, gts):
        if isinstance(pred_raw, str):
            pred_json = json.loads(pred_raw, strict=False)
        else:
            pred_json = pred_raw

        gt_fields = [Field(label, value) for label, value in gt_json.items()]
        pred_fields = [Field(label, pred_json.get(label, "")) for label in gt_json.keys()]

        prediction = Prediction(fields=pred_fields, gt=GroundTruth(gt_fields))
        prediction_objects.append(prediction)

    return get_kie_metrics(prediction_objects)

 
def run_inference(image, model, tokenizer, instruction):
    """
    Runs inference on `image` + `instruction` through `model`/`tokenizer`,
    and returns:
      - generated_caption (str)
      - inference_time_s (float)
      - peak_vram_mb (float)
    """
    try:
        # Build the chat prompt
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        print(f"📝 Tokenized prompt: {input_text[:100]}...")

        # Prepare inputs and move to CUDA
        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        inputs.pop("token_type_ids", None)

        # Reset CUDA memory stats so we can track this run’s peak
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
                "max_new_tokens": 128,
                "use_cache": True,
                "temperature": 1.0,
                "min_p": 0.1
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

        return generated_caption.strip(), inference_time_s, peak_vram_mb

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        # On error, return empty caption and zeros
        return "", 0.0, 0.0

def evaluate_batch(prompt, val_data, indexes, multiple_refs=True, MODEL_DIR="/workspace/unsloth-finetune", LOAD_FROM_HF=False):
    """
    prompts_list: list of instructions to evaluate
    val_data: DataFrame with ['image', 'caption'] columns,
    indexes: list of indexes to sample from val_data
    """
    print(f"🔄 Loading vision-language model from {MODEL_DIR}...")
    BASE_MODEL = "unsloth/Qwen2-VL-7B-Instruct"  
    # --- Load model ---
    if LOAD_FROM_HF:
        print(f"🔄 Loading base model '{BASE_MODEL}'...")
        model, tokenizer = FastVisionModel.from_pretrained(
            BASE_MODEL,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )

        print(f"🔄 Applying adapter from '{MODEL_DIR}'...")
        model.load_adapter(MODEL_DIR)
    else:
        print(f"🔄 Loading full model directly from '{MODEL_DIR}'...")
        model, tokenizer = FastVisionModel.from_pretrained(
            MODEL_DIR,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )

    model.eval()
    print("✅ Model loaded successfully.")

    scorer = SentenceTransformer("all-MiniLM-L6-v2").to("cuda")
    print("✅ Sentence transformer loaded.")

    print("🚀 Starting batch evaluation...")
    all_results = {}
    Inference_time = {}
    Vram_usages = {}
    gts = []
    res = []
    for index in indexes: 
        print(f"\n📦 Evaluating sample {index+1}/{len(indexes)} at index {index}...")
        sample = val_data[index]
        if multiple_refs:
            reference_list = sample['caption'] 
            pred, inference_time, peak_vram = run_inference(sample['image'], model, tokenizer, prompt)

        else:
            reference_list = [sample['caption']]
            pred, inference_time, peak_vram = run_inference(sample['image'], model, tokenizer, prompt)
        res.append(pred)                                # list of prediction strings
        gts.append({"caption": reference_list})          # list of dicts with "caption" key 
        all_results[index] = pred
        Inference_time[index] = inference_time
        Vram_usages[index] = peak_vram

    avg_score, per_sample_scores = evaluate_kie_predictions(res,gts)
       
    print("✅ Batch evaluation complete!")
    return all_results, per_sample_scores, Inference_time, Vram_usages

