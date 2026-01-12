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


import json
from typing import List, Dict, Tuple

def evaluate_kie_totals(preds: List[str], gts: List[Dict]) -> Tuple[
    List[int],  # summed_total == bill_total
    List[int],  # bill_total == ground_truth_total
    List[int]   # summed_total == ground_truth_total
]:
    """
    Evaluate KIE predictions focusing on totals.

    For each prediction:
      1. Identify a value with 'total' in the key (bill_total).
      2. Remove all 'total' fields from prediction.
      3. Assert remaining values are numbers, sum them (summed_total).
      4. From ground truth, identify 'total_amount' (gt_total).
      5. Compute three binary indicators:
         - summed_total == bill_total
         - bill_total == gt_total
         - summed_total == gt_total

    Args:
        preds: List of prediction strings (JSON format or dicts)
        gts: List of ground truth dicts (JSON or dicts)

    Returns:
        Tuple of three lists of binary values (0/1) for each sample.
    """
    assert len(preds) == len(gts), "Predictions and ground truth lists must be the same length."

    eq_sum_vs_bill = []
    eq_bill_vs_gt = []
    eq_sum_vs_gt = []

    for pred_raw, gt_raw in zip(preds, gts):
        # Parse prediction
        if isinstance(pred_raw, str):
            try:
                pred_json = json.loads(pred_raw)
            except json.JSONDecodeError:
                pred_json = {}
        else:
            pred_json = pred_raw

        # Parse ground truth
        if isinstance(gt_raw, str):
            try:
                gt_json = json.loads(gt_raw)
            except json.JSONDecodeError:
                gt_json = {}
        else:
            gt_json = gt_raw

        # 1. Identify bill_total (any key containing 'total')
        bill_total = None
        for k, v in pred_json.items():
            if "total" in k.lower():
                try:
                    bill_total = float(v)
                except (ValueError, TypeError):
                    bill_total = None
                break

        # 2. Remove all 'total' fields from prediction
        filtered_pred = {k: v for k, v in pred_json.items() if "total" not in k.lower()}

        # 3. Assert remaining values are numbers and sum them
        summed_total = 0.0
        for v in filtered_pred.values():
            try:
                summed_total += float(v)
            except (ValueError, TypeError):
                continue  # Ignore non-numeric values
        # 4. Identify ground truth total_amount
        gt_total = None
        for k, v in gt_json.items():
            if k.lower() == "total_amount":
                try:
                    gt_total = float(v)
                except (ValueError, TypeError):
                    gt_total = None
                break

        # 5. Compute binary indicators
        eq_sum_vs_bill.append(int(summed_total == bill_total if bill_total is not None else 0))
        eq_bill_vs_gt.append(int(bill_total == gt_total if bill_total is not None and gt_total is not None else 0))
        eq_sum_vs_gt.append(int(summed_total == gt_total if gt_total is not None else 0))

    return eq_sum_vs_bill, eq_bill_vs_gt, eq_sum_vs_gt

def log_metrics_to_excel(
    sample_indices: List[int],
    predictions: Dict[int, str],
    ground_truths: Dict[int, str],
    eq_sum_vs_bill: List[int],
    eq_bill_vs_gt: List[int],
    eq_sum_vs_gt: List[int],
    inference_times: Dict[int, float],
    vram_usage: Dict[int, float],
    test_dataset,
    prompt: str,
    model_name: str,
    output_excel_path: str = "KIE_results.xlsx"
):
    """
    Log evaluation metrics (binary totals checks) to an Excel file with images.
    """
    rows = []
    pil_images = []

    for i, idx in enumerate(sample_indices):
        pred = predictions.get(idx, "")
        gt = ground_truths.get(idx, "")
        time_taken = inference_times.get(idx, 0.0)
        vram = vram_usage.get(idx, 0.0)

        sum_vs_bill = eq_sum_vs_bill[i] if i < len(eq_sum_vs_bill) else 0
        bill_vs_gt = eq_bill_vs_gt[i] if i < len(eq_bill_vs_gt) else 0
        sum_vs_gt = eq_sum_vs_gt[i] if i < len(eq_sum_vs_gt) else 0

        # Get image from dataset
        sample_item = test_dataset[idx]
        pil_img = sample_item['image']

        if isinstance(pil_img, bytes):
            pil_img = PILImage.open(BytesIO(pil_img))
        elif not isinstance(pil_img, PILImage.Image):
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
            "eq_sum_vs_bill": sum_vs_bill,
            "eq_bill_vs_gt": bill_vs_gt,
            "eq_sum_vs_gt": sum_vs_gt,
            "inference_time_s": time_taken,
            "vram_usage_mb": vram,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    writer = pd.ExcelWriter(output_excel_path, engine="xlsxwriter")
    sheet_name = "evaluation"
    df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=1, index=False)

    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    for row_idx, pil_img in enumerate(pil_images, start=1):
        img_stream = BytesIO()
        pil_img.thumbnail((128, 128))
        pil_img.save(img_stream, format="PNG")
        img_stream.seek(0)
        worksheet.insert_image(row_idx, 0, f"image_{row_idx}.png", {"image_data": img_stream})

    worksheet.set_column(0, 0, 20)
    worksheet.set_column(1, 1, 12)
    worksheet.set_column(2, 2, 50)
    worksheet.set_column(3, 3, 20)
    worksheet.set_column(4, 4, 40)
    worksheet.set_column(5, 5, 40)
    worksheet.set_column(6, 8, 15)  # binary metrics
    worksheet.set_column(9, 9, 15)
    worksheet.set_column(10, 10, 15)

    writer.close()

    print(f"✅ Excel written to: {output_excel_path}")

    # Summary stats
    avg_sum_vs_bill = sum(eq_sum_vs_bill) / len(eq_sum_vs_bill) if eq_sum_vs_bill else 0.0
    avg_bill_vs_gt = sum(eq_bill_vs_gt) / len(eq_bill_vs_gt) if eq_bill_vs_gt else 0.0
    avg_sum_vs_gt = sum(eq_sum_vs_gt) / len(eq_sum_vs_gt) if eq_sum_vs_gt else 0.0
    avg_time = sum(inference_times.values()) / len(inference_times) if inference_times else 0.0
    avg_vram = sum(vram_usage.values()) / len(vram_usage) if vram_usage else 0.0

    print(f"\n📊 Summary Statistics:")
    print(f"   - % Sum == Bill Total: {avg_sum_vs_bill:.2%}")
    print(f"   - % Bill Total == Ground Truth: {avg_bill_vs_gt:.2%}")
    print(f"   - % Sum == Ground Truth: {avg_sum_vs_gt:.2%}")
    print(f"   - Average Inference Time: {avg_time:.3f}s")
    print(f"   - Average VRAM Usage: {avg_vram:.2f} MB")

    return df

def log_metrics_to_wandb(
    sample_indices: List[int],
    predictions: Dict[int, str],
    ground_truths: Dict[int, str],
    eq_sum_vs_bill: List[int],
    eq_bill_vs_gt: List[int],
    eq_sum_vs_gt: List[int],
    inference_times: Dict[int, float],
    vram_usage: Dict[int, float],
    test_dataset,
    prompt: str,
    model_name: str,
    wandb_entity: str = None,
    wandb_project: str = "kie-experiments",
    wandb_run_name: str = None
):
    """
    Log binary totals evaluation metrics to WandB.
    """
    try:
        import wandb
    except ImportError:
        print("⚠️  WandB not installed. Skipping WandB logging.")
        return

    try:
        wandb.init(entity=wandb_entity, project=wandb_project, name=wandb_run_name, reinit=True)

        # Summary metrics
        avg_sum_vs_bill = sum(eq_sum_vs_bill) / len(eq_sum_vs_bill) if eq_sum_vs_bill else 0.0
        avg_bill_vs_gt = sum(eq_bill_vs_gt) / len(eq_bill_vs_gt) if eq_bill_vs_gt else 0.0
        avg_sum_vs_gt = sum(eq_sum_vs_gt) / len(eq_sum_vs_gt) if eq_sum_vs_gt else 0.0
        avg_time = sum(inference_times.values()) / len(inference_times) if inference_times else 0.0
        avg_vram = sum(vram_usage.values()) / len(vram_usage) if vram_usage else 0.0

        wandb.log({
            "avg_sum_vs_bill": avg_sum_vs_bill,
            "avg_bill_vs_gt": avg_bill_vs_gt,
            "avg_sum_vs_gt": avg_sum_vs_gt,
            "avg_inference_time_s": avg_time,
            "avg_vram_usage_mb": avg_vram,
        })

        # Table
        table_cols = ["sample_index", "image", "prediction", "ground_truth",
                      "eq_sum_vs_bill", "eq_bill_vs_gt", "eq_sum_vs_gt",
                      "inference_time_s", "vram_usage_mb", "prompt"]
        wandb_table = wandb.Table(columns=table_cols)

        for i, idx in enumerate(sample_indices):
            pred = predictions.get(idx, "")
            gt = ground_truths.get(idx, "")
            time_taken = inference_times.get(idx, 0.0)
            vram = vram_usage.get(idx, 0.0)

            sum_vs_bill = eq_sum_vs_bill[i] if i < len(eq_sum_vs_bill) else 0
            bill_vs_gt = eq_bill_vs_gt[i] if i < len(eq_bill_vs_gt) else 0
            sum_vs_gt = eq_sum_vs_gt[i] if i < len(eq_sum_vs_gt) else 0

            sample_item = test_dataset[idx]
            pil_img = sample_item['image']

            if isinstance(pil_img, bytes):
                pil_img = PILImage.open(BytesIO(pil_img))
            elif not isinstance(pil_img, PILImage.Image):
                pil_img = PILImage.fromarray(pil_img)

            wb_image = None
            try:
                wb_image = wandb.Image(pil_img, caption=f"sample_{idx}")
            except Exception:
                wb_image = None

            if isinstance(gt, dict):
                gt_str = json.dumps(gt, indent=2)
            else:
                gt_str = str(gt)

            wandb_table.add_data(idx, wb_image, pred, gt_str,
                                 sum_vs_bill, bill_vs_gt, sum_vs_gt,
                                 time_taken, vram, prompt)

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
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="WandB entity/team name")
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

    # Evaluate with new binary totals function
    print("\n🔄 Evaluating predictions...")
    eq_sum_vs_bill, eq_bill_vs_gt, eq_sum_vs_gt = evaluate_kie_totals(pred_list, gt_list)

    print(f"✅ Evaluation complete!")
    print(f"   - % Sum == Bill Total: {sum(eq_sum_vs_bill)/len(eq_sum_vs_bill):.2%}")
    print(f"   - % Bill Total == Ground Truth: {sum(eq_bill_vs_gt)/len(eq_bill_vs_gt):.2%}")
    print(f"   - % Sum == Ground Truth: {sum(eq_sum_vs_gt)/len(eq_sum_vs_gt):.2%}")

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
        eq_sum_vs_bill,
        eq_bill_vs_gt,
        eq_sum_vs_gt,
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
            eq_sum_vs_bill,
            eq_bill_vs_gt,
            eq_sum_vs_gt,
            inference_times,
            vram_usage,
            test_dataset,
            prompt,
            args.model_name,
            args.wandb_entity,
            args.wandb_project,
            args.wandb_run_name
        )

    print("\n✅ Evaluation complete!")
