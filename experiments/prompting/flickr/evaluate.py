#!/usr/bin/env python3
"""
Evaluate Flickr30k predictions using **official COCO pycocoevalcap CIDEr** scorer.
(Uses shared global IDF computed from all references — the standard method.)

Install:
    pip install git+https://github.com/salaniz/pycocoevalcap
    pip install pandas pillow xlsxwriter tqdm datasets

"""

from __future__ import annotations
import argparse
import json
from io import BytesIO
from typing import List, Dict, Tuple
import pandas as pd
from PIL import Image as PILImage
from tqdm import tqdm

# ---- pycocoevalcap (official COCO) ----
from pycocoevalcap.cider.cider import Cider

# ---- Preprocess ----
def preprocess_caption(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.strip().lower()

# ---- CIDEr computation ----
def compute_cider_scores(predictions: List[str], references: List[List[str]]) -> Tuple[float, List[float]]:
    """Compute CIDEr using official pycocoevalcap implementation."""
    gts = {}
    res = {}

    for i, (pred, ref_list) in enumerate(zip(predictions, references)):
        pred_pp = preprocess_caption(pred)
        ref_pp = [preprocess_caption(r) for r in ref_list]

        res[i] = [{"caption": pred_pp}]
        gts[i] = [{"caption": r} for r in ref_pp]

    cider_scorer = Cider()
    cider_score, scores = cider_scorer.compute_score(gts, res)

    return float(cider_score), [float(s) for s in scores]

# ---- Excel Logging ----
def log_metrics_to_excel(
    sample_indices: List[int],
    predictions: Dict[int, str],
    ground_truths: Dict[int, List[str]],
    cider_scores: List[float],
    inference_times: Dict[int, float],
    vram_usage: Dict[int, float],
    test_dataset,
    prompt: str,
    model_name: str,
    output_excel_path: str = "flickr30k_cider_results.xlsx",
):
    rows = []
    pil_images = []

    for i, idx in enumerate(sample_indices):
        pred = predictions.get(idx, "")
        gt_list = ground_truths.get(idx, [])
        time_taken = inference_times.get(idx, 0.0)
        vram = vram_usage.get(idx, 0.0)
        cider = cider_scores[i] if i < len(cider_scores) else 0.0

        # Load image
        sample_item = test_dataset[idx]
        pil_img = sample_item.get("image") if isinstance(sample_item, dict) else sample_item["image"]

        if isinstance(pil_img, bytes):
            pil_img = PILImage.open(BytesIO(pil_img))
        elif not isinstance(pil_img, PILImage.Image):
            try:
                import numpy as _np
                pil_img = PILImage.fromarray(_np.asarray(pil_img))
            except Exception:
                pil_img = None

        pil_images.append(pil_img)

        gt_text = "\n".join(str(c) for c in gt_list)

        row = {
            "sample_index": idx,
            "prompt": prompt,
            "model": model_name,
            "ground_truths": gt_text,
            "prediction": pred,
            "cider_score": cider,
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

    # Insert images on column A
    for row_idx, pil_img in enumerate(pil_images, start=1):
        if pil_img is None:
            continue
        try:
            stream = BytesIO()
            pil_img.thumbnail((128, 128))
            pil_img.save(stream, format="PNG")
            stream.seek(0)
            worksheet.insert_image(row_idx, 0, f"img_{row_idx}.png", {"image_data": stream})
        except Exception:
            pass

    worksheet.set_column(0, 0, 20)
    worksheet.set_column(1, 1, 12)
    worksheet.set_column(2, 2, 50)
    worksheet.set_column(3, 3, 20)
    worksheet.set_column(4, 4, 60)
    worksheet.set_column(5, 5, 50)
    worksheet.set_column(6, 6, 12)
    worksheet.set_column(7, 7, 15)
    worksheet.set_column(8, 8, 15)

    writer.close()
    print(f"✅ Excel written to: {output_excel_path}")

# ---- WandB Logging ----
def log_metrics_to_wandb(
    sample_indices,
    predictions,
    ground_truths,
    cider_scores,
    inference_times,
    vram_usage,
    test_dataset,
    prompt,
    model_name,
    project_name,
    run_name=None,
):
    try:
        import wandb
    except ImportError:
        print("⚠️ WandB not installed — skipping.")
        return

    run = wandb.init(project=project_name, name=run_name, reinit=True)

    table = wandb.Table(columns=[
        "sample_index", "image", "prediction", "ground_truths",
        "cider_score", "inference_time_s", "vram_usage_mb", "prompt"
    ])

    for i, idx in enumerate(sample_indices):
        pred = predictions.get(idx, "")
        gt_list = ground_truths.get(idx, [])
        gt_str = "\n".join(gt_list)
        cider = cider_scores[i]

        # image
        sample_item = test_dataset[idx]
        pil_img = sample_item.get("image") if isinstance(sample_item, dict) else sample_item["image"]

        if isinstance(pil_img, bytes):
            pil_img = PILImage.open(BytesIO(pil_img))
        elif not isinstance(pil_img, PILImage.Image):
            try:
                import numpy as _np
                pil_img = PILImage.fromarray(_np.asarray(pil_img))
            except Exception:
                pil_img = None

        wb_img = wandb.Image(pil_img) if pil_img else None

        table.add_data(
            idx, wb_img, pred, gt_str, cider,
            inference_times.get(idx, 0.0),
            vram_usage.get(idx, 0.0),
            prompt,
        )

    wandb.log({"evaluation_table": table})
    wandb.finish()
    print("✅ Logged to WandB")

# ---- MAIN ----
if __name__ == "__main__":
    import datasets

    parser = argparse.ArgumentParser(description="Evaluate Flickr30k predictions using CIDEr")

    # Input
    parser.add_argument("--inference-results", type=str, required=True,
                        help="Path to inference results JSON file")
    parser.add_argument("--test-dataset", type=str, required=True,
                        help="Path to Flickr30k test dataset")

    # Output
    parser.add_argument("--output-excel", type=str, default="flickr30k_cider_results.xlsx",
                        help="Output Excel file path")
    parser.add_argument("--model-name", type=str, default="my-model",
                        help="Model name for logging")

    # WandB
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="flickr30k-eval",
                        help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="WandB run name (optional)")

    args = parser.parse_args()

    # Load inference results
    print(f"🔄 Loading inference results from: {args.inference_results}")
    with open(args.inference_results, "r") as f:
        results = json.load(f)

    predictions = results.get("predictions", {})
    ground_truths = results.get("ground_truths", {})
    inference_times = results.get("inference_times", {})
    vram_usage = results.get("vram_usage", {})
    sample_indices = results.get("sample_indices", [])
    prompt = results.get("prompt", "")

    # Convert string keys to int for dictionaries
    predictions = {int(k): v for k, v in predictions.items()}
    ground_truths = {int(k): v for k, v in ground_truths.items()}
    inference_times = {int(k): v for k, v in inference_times.items()}
    vram_usage = {int(k): v for k, v in vram_usage.items()}

    # Prepare lists for evaluation
    pred_list = [predictions[idx] for idx in sample_indices]
    gt_list = [ground_truths[idx] for idx in sample_indices]

    # Evaluate using CIDEr
    print("\n🔄 Computing CIDEr scores...")
    corpus_cider, per_sample_cider = compute_cider_scores(pred_list, gt_list)

    print(f"✅ Evaluation complete!")
    print(f"   - CIDEr (corpus): {corpus_cider:.4f}")

    # Load test dataset for images
    print(f"\n🔄 Loading test dataset from: {args.test_dataset}")
    test_dataset = datasets.load_from_disk(args.test_dataset)

    # Log to Excel
    print(f"\n💾 Logging to Excel...")
    log_metrics_to_excel(
        sample_indices,
        predictions,
        ground_truths,
        per_sample_cider,
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
            per_sample_cider,
            inference_times,
            vram_usage,
            test_dataset,
            prompt,
            args.model_name,
            args.wandb_project,
            args.wandb_run_name
        )

    print("\n✅ Evaluation complete!")
