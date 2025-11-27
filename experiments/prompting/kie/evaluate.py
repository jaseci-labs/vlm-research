#!/usr/bin/env python3
"""
Evaluate KIE predictions using Levenshtein edit distance.
Logs results to Excel and optionally to WandB.
"""

import argparse
import json
import os
from typing import List, Dict, Any
import pandas as pd
from io import BytesIO
from PIL import Image as PILImage
from Levenshtein import distance as edit_distance
from tqdm import tqdm


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

def evaluate_kie_predictions(preds: List[str], gts: List[Dict]) -> tuple[float, List[float]]:
    """
    Evaluate KIE predictions against ground truth using Levenshtein similarity.

    Args:
        preds: List of prediction strings (JSON format)
        gts: List of ground truth dicts

    Returns:
        - Average accuracy score across all samples.
        - List of individual sample-level accuracy scores.
    """
    assert len(preds) == len(gts), "Predictions and ground truth lists must be the same length."

    prediction_objects = []

    for pred_raw, gt_json in zip(preds, gts):
        # Parse prediction if it's a string
        if isinstance(pred_raw, str):
            try:
                pred_json = json.loads(pred_raw)
            except json.JSONDecodeError:
                # If parsing fails, treat as empty prediction
                pred_json = {}
        else:
            pred_json = pred_raw

        # Parse ground truth if it's a string
        if isinstance(gt_json, str):
            try:
                gt_json = json.loads(gt_json)
            except json.JSONDecodeError:
                gt_json = {}

        gt_fields = [Field(label, value) for label, value in gt_json.items()]
        pred_fields = [Field(label, pred_json.get(label, "")) for label in gt_json.keys()]

        prediction = Prediction(fields=pred_fields, gt=GroundTruth(gt_fields))
        prediction_objects.append(prediction)

    return get_kie_metrics(prediction_objects)


# ----- Excel logging -----

def log_metrics_to_excel(
    sample_indices: List[int],
    predictions: Dict[int, str],
    ground_truths: Dict[int, str],
    kie_scores: List[float],
    inference_times: Dict[int, float],
    vram_usage: Dict[int, float],
    test_dataset,
    prompt: str,
    model_name: str,
    output_excel_path: str = "KIE_results.xlsx"
):
    """
    Log evaluation metrics to an Excel file with images.

    Args:
        sample_indices: List of sample indices
        predictions: Dictionary of predictions
        ground_truths: Dictionary of ground truths
        kie_scores: List of KIE scores per sample
        inference_times: Dictionary of inference times
        vram_usage: Dictionary of VRAM usage
        test_dataset: Test dataset
        prompt: Prompt used
        model_name: Model name
        output_excel_path: Path to save Excel file
    """
    rows = []
    pil_images = []

    for i, idx in enumerate(sample_indices):
        pred = predictions.get(idx, "")
        gt = ground_truths.get(idx, "")
        time_taken = inference_times.get(idx, 0.0)
        vram = vram_usage.get(idx, 0.0)
        kie = kie_scores[i] if i < len(kie_scores) else 0.0

        # Get image from dataset
        sample_item = test_dataset[idx]
        pil_img = sample_item['image']

        # Handle different image types: bytes, numpy array, or PIL Image
        if isinstance(pil_img, bytes):
            # Decode bytes to PIL Image
            pil_img = PILImage.open(BytesIO(pil_img))
        elif not isinstance(pil_img, PILImage.Image):
            # Convert numpy array to PIL Image
            pil_img = PILImage.fromarray(pil_img)

        pil_images.append(pil_img)

        # Format ground truth
        if isinstance(gt, dict):
            caption_text = json.dumps(gt, indent=2)
        elif isinstance(gt, list):
            caption_text = "\n".join(str(c) for c in gt)
        else:
            caption_text = str(gt)

        row = {
            "sample_index": idx,
            "prompt": prompt,
            "model": model_name,
            "ground_truth": caption_text,
            "prediction": pred,
            "kie_score": kie,
            "inference_time_s": time_taken,
            "vram_usage_mb": vram,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Write DataFrame to Excel
    writer = pd.ExcelWriter(output_excel_path, engine="xlsxwriter")
    sheet_name = "evaluation"
    df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=1, index=False)

    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    # Insert images into column A
    for row_idx, pil_img in enumerate(pil_images, start=1):
        img_stream = BytesIO()
        pil_img.thumbnail((128, 128))
        pil_img.save(img_stream, format="PNG")
        img_stream.seek(0)
        worksheet.insert_image(row_idx, 0, f"image_{row_idx}.png", {"image_data": img_stream})

    # Set column widths
    worksheet.set_column(0, 0, 20)  # Image column
    worksheet.set_column(1, 1, 12)  # sample_index
    worksheet.set_column(2, 2, 50)  # prompt
    worksheet.set_column(3, 3, 20)  # model
    worksheet.set_column(4, 4, 40)  # ground_truth
    worksheet.set_column(5, 5, 40)  # prediction
    worksheet.set_column(6, 6, 12)  # kie_score
    worksheet.set_column(7, 7, 15)  # inference_time_s
    worksheet.set_column(8, 8, 15)  # vram_usage_mb

    writer.close()

    print(f"✅ Excel written to: {output_excel_path}")

    # Calculate summary stats
    avg_kie = sum(kie_scores) / len(kie_scores) if kie_scores else 0.0
    avg_time = sum(inference_times.values()) / len(inference_times) if inference_times else 0.0
    avg_vram = sum(vram_usage.values()) / len(vram_usage) if vram_usage else 0.0

    print(f"\n📊 Summary Statistics:")
    print(f"   - Average KIE Score: {avg_kie:.4f}")
    print(f"   - Average Inference Time: {avg_time:.3f}s")
    print(f"   - Average VRAM Usage: {avg_vram:.2f} MB")

    return df


# ----- WandB logging (optional) -----

def log_metrics_to_wandb(
    sample_indices: List[int],
    predictions: Dict[int, str],
    ground_truths: Dict[int, str],
    kie_scores: List[float],
    inference_times: Dict[int, float],
    vram_usage: Dict[int, float],
    test_dataset,
    prompt: str,
    model_name: str,
    wandb_project: str = "kie-experiments",
    wandb_run_name: str = None
):
    """
    Log evaluation metrics to WandB.

    Args:
        sample_indices: List of sample indices
        predictions: Dictionary of predictions
        ground_truths: Dictionary of ground truths
        kie_scores: List of KIE scores per sample
        inference_times: Dictionary of inference times
        vram_usage: Dictionary of VRAM usage
        test_dataset: Test dataset
        prompt: Prompt used
        model_name: Model name
        wandb_project: WandB project name
        wandb_run_name: WandB run name (optional)
    """
    try:
        import wandb
    except ImportError:
        print("⚠️  WandB not installed. Skipping WandB logging.")
        return

    try:
        run = wandb.init(project=wandb_project, name=wandb_run_name, reinit=True)

        # Log summary metrics
        avg_kie = sum(kie_scores) / len(kie_scores) if kie_scores else 0.0
        avg_time = sum(inference_times.values()) / len(inference_times) if inference_times else 0.0
        avg_vram = sum(vram_usage.values()) / len(vram_usage) if vram_usage else 0.0

        wandb.log({
            "avg_kie_score": avg_kie,
            "avg_inference_time_s": avg_time,
            "avg_vram_usage_mb": avg_vram,
        })

        # Create table
        table_cols = ["sample_index", "image", "prediction", "ground_truth",
                      "kie_score", "inference_time_s", "vram_usage_mb", "prompt"]
        wandb_table = wandb.Table(columns=table_cols)

        for i, idx in enumerate(sample_indices):
            pred = predictions.get(idx, "")
            gt = ground_truths.get(idx, "")
            time_taken = inference_times.get(idx, 0.0)
            vram = vram_usage.get(idx, 0.0)
            kie = kie_scores[i] if i < len(kie_scores) else 0.0

            # Get image
            sample_item = test_dataset[idx]
            pil_img = sample_item['image']

            # Handle different image types: bytes, numpy array, or PIL Image
            if isinstance(pil_img, bytes):
                # Decode bytes to PIL Image
                pil_img = PILImage.open(BytesIO(pil_img))
            elif not isinstance(pil_img, PILImage.Image):
                # Convert numpy array to PIL Image
                pil_img = PILImage.fromarray(pil_img)

            wb_image = None
            try:
                wb_image = wandb.Image(pil_img, caption=f"sample_{idx}")
            except Exception:
                wb_image = None

            # Format ground truth
            if isinstance(gt, dict):
                gt_str = json.dumps(gt, indent=2)
            else:
                gt_str = str(gt)

            wandb_table.add_data(idx, wb_image, pred, gt_str, kie, time_taken, vram, prompt)

        wandb.log({"evaluation_table": wandb_table})
        wandb.finish()

        print(f"✅ Metrics logged to WandB project: {wandb_project}")

    except Exception as e:
        print(f"⚠️  Failed to log to WandB: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KIE predictions")

    # Input
    parser.add_argument("--inference-results", type=str, required=True,
                        help="Path to inference results JSON file")
    parser.add_argument("--test-dataset", type=str, default="./kie_splits/test",
                        help="Path to test dataset")

    # Output
    parser.add_argument("--output-excel", type=str, default="KIE_results.xlsx",
                        help="Output Excel file path")
    parser.add_argument("--model-name", type=str, default="gemma-3-12b-it",
                        help="Model name for logging")

    # WandB
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="kie-eval",
                        help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="WandB run name (optional)")

    args = parser.parse_args()

    # Load inference results
    print(f"🔄 Loading inference results from: {args.inference_results}")
    with open(args.inference_results, "r") as f:
        results = json.load(f)

    predictions = results["predictions"]
    ground_truths = results["ground_truths"]
    inference_times = results["inference_times"]
    vram_usage = results["vram_usage"]
    sample_indices = results["sample_indices"]
    prompt = results["prompt"]

    # Convert string keys to int for dictionaries
    predictions = {int(k): v for k, v in predictions.items()}
    ground_truths = {int(k): v for k, v in ground_truths.items()}
    inference_times = {int(k): v for k, v in inference_times.items()}
    vram_usage = {int(k): v for k, v in vram_usage.items()}

    # Prepare lists for evaluation
    pred_list = [predictions[idx] for idx in sample_indices]
    gt_list = [ground_truths[idx] for idx in sample_indices]

    # Evaluate
    print("\n🔄 Evaluating predictions...")
    avg_score, per_sample_scores = evaluate_kie_predictions(pred_list, gt_list)

    print(f"✅ Evaluation complete!")
    print(f"   - Average KIE Score: {avg_score:.4f}")

    # Load test dataset for images
    from datasets import load_from_disk
    print(f"\n🔄 Loading test dataset from: {args.test_dataset}")
    test_dataset = load_from_disk(args.test_dataset)

    # Log to Excel
    print(f"\n💾 Logging to Excel...")
    log_metrics_to_excel(
        sample_indices,
        predictions,
        ground_truths,
        per_sample_scores,
        inference_times,
        vram_usage,
        test_dataset,
        prompt,
        args.model_name,
        args.output_excel
    )

    # Log to WandB if enabled
    if args.use_wandb:
        print(f"\n💾 Logging to WandB...")
        log_metrics_to_wandb(
            sample_indices,
            predictions,
            ground_truths,
            per_sample_scores,
            inference_times,
            vram_usage,
            test_dataset,
            prompt,
            args.model_name,
            args.wandb_project,
            args.wandb_run_name
        )

    print("\n✅ Evaluation complete!")
