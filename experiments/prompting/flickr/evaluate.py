#!/usr/bin/env python3
"""
Evaluate Flickr30k predictions using CIDEr, SPICE, and Cosine similarity metrics.

- Computes average score over 5 references by default.
- Optionally, compute maximum score per sample using `--use-max-ref`.
- Logs results to Excel and optionally to WandB.

Install:
    pip install git+https://github.com/salaniz/pycocoevalcap
    pip install pandas pillow xlsxwriter tqdm datasets sentence-transformers
"""

from __future__ import annotations
import argparse
import json
from io import BytesIO
from typing import List, Dict, Tuple
import pandas as pd
from PIL import Image as PILImage
from tqdm import tqdm

# ---- pycocoevalcap ----
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# ---- sentence-transformers for Cosine similarity ----
from sentence_transformers import SentenceTransformer, util

# ---- Preprocess ----
def preprocess_caption(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.strip().lower()

# ---- CIDEr computation ----
def compute_cider_scores(predictions: List[str], references: List[List[str]], average_over_refs: bool = True) -> Tuple[float, List[float]]:
    gts, res = {}, {}
    for i, (pred, ref_list) in enumerate(zip(predictions, references)):
        pred_pp = preprocess_caption(pred)
        ref_pp = [preprocess_caption(r) for r in ref_list]
        res[i] = [{"caption": pred_pp}]
        gts[i] = [{"caption": r} for r in ref_pp]

    cider_scorer = Cider()
    corpus_score, per_sample_scores = cider_scorer.compute_score(gts, res)

    if average_over_refs:
        return float(corpus_score), [float(s) for s in per_sample_scores]
    else:
        # Compute max per sample
        per_sample_max = []
        for i, ref_list in enumerate(references):
            scores = []
            for r in ref_list:
                s, _ = cider_scorer.compute_score({i:[{"caption": r}]}, {i:[{"caption": preprocess_caption(predictions[i])}]})
                scores.append(s)
            per_sample_max.append(max(scores))
        return float(corpus_score), per_sample_max

# ---- SPICE computation ----
def compute_spice_scores(predictions: List[str], references: List[List[str]], average_over_refs: bool = True) -> Tuple[float, List[float]]:
    gts, res = {}, {}
    for i, (pred, ref_list) in enumerate(zip(predictions, references)):
        pred_pp = preprocess_caption(pred)
        ref_pp = [preprocess_caption(r) for r in ref_list]
        res[i] = [{"caption": pred_pp}]
        gts[i] = [{"caption": r} for r in ref_pp]

    spice_scorer = Spice()
    corpus_score, per_sample_scores = spice_scorer.compute_score(gts, res)

    if average_over_refs:
        return float(corpus_score), [float(s) for s in per_sample_scores]
    else:
        # Compute max per sample
        per_sample_max = []
        for i, ref_list in enumerate(references):
            scores = []
            for r in ref_list:
                s, _ = spice_scorer.compute_score({i:[{"caption": r}]}, {i:[{"caption": preprocess_caption(predictions[i])}]})
                scores.append(s)
            per_sample_max.append(max(scores))
        return float(corpus_score), per_sample_max

# ---- Cosine similarity computation ----
def compute_cosine_scores(predictions: List[str], references: List[List[str]], average_over_refs: bool = True) -> Tuple[float, List[float]]:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    per_sample_scores = []

    for i, (pred, ref_list) in enumerate(zip(predictions, references)):
        pred_emb = model.encode(pred, convert_to_tensor=True)
        ref_embs = model.encode(ref_list, convert_to_tensor=True)
        sims = util.cos_sim(pred_emb, ref_embs).cpu().numpy().flatten()
        if average_over_refs:
            per_sample_scores.append(float(sims.mean()))
        else:
            per_sample_scores.append(float(sims.max()))

    corpus_score = sum(per_sample_scores) / len(per_sample_scores)
    return corpus_score, per_sample_scores

# ---- Excel Logging ----
def log_metrics_to_excel(
    sample_indices: List[int],
    predictions: Dict[int, str],
    ground_truths: Dict[int, List[str]],
    cider_scores: List[float],
    spice_scores: List[float],
    cosine_scores: List[float],
    inference_times: Dict[int, float],
    vram_usage: Dict[int, float],
    test_dataset,
    prompt: str,
    model_name: str,
    output_excel_path: str = "flickr30k_eval_results.xlsx",
):
    rows = []
    pil_images = []

    for i, idx in enumerate(sample_indices):
        pred = predictions.get(idx, "")
        gt_list = ground_truths.get(idx, [])
        time_taken = inference_times.get(idx, 0.0)
        vram = vram_usage.get(idx, 0.0)
        cider = cider_scores[i] if i < len(cider_scores) else 0.0
        spice = spice_scores[i] if i < len(spice_scores) else 0.0
        cosine = cosine_scores[i] if i < len(cosine_scores) else 0.0

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
            "spice_score": spice,
            "cosine_score": cosine,
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
    worksheet.set_column(5, 5, 15)
    worksheet.set_column(6, 6, 15)
    worksheet.set_column(7, 7, 15)
    worksheet.set_column(8, 8, 12)
    worksheet.set_column(9, 9, 15)

    writer.close()
    print(f"✅ Excel written to: {output_excel_path}")

# ---- WandB Logging ----
def log_metrics_to_wandb(
    sample_indices,
    predictions,
    ground_truths,
    cider_scores,
    spice_scores,
    cosine_scores,
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
        "cider_score", "spice_score", "cosine_score",
        "inference_time_s", "vram_usage_mb", "prompt"
    ])

    for i, idx in enumerate(sample_indices):
        pred = predictions.get(idx, "")
        gt_list = ground_truths.get(idx, [])
        gt_str = "\n".join(gt_list)
        cider = cider_scores[i]
        spice = spice_scores[i]
        cosine = cosine_scores[i]

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
            idx, wb_img, pred, gt_str, cider, spice, cosine,
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

    parser = argparse.ArgumentParser(description="Evaluate Flickr30k predictions using CIDEr, SPICE, and Cosine similarity")

    # Input
    parser.add_argument("--inference-results", type=str, required=True,
                        help="Path to inference results JSON file")
    parser.add_argument("--test-dataset", type=str, required=True,
                        help="Path to Flickr30k test dataset")

    # Output
    parser.add_argument("--output-excel", type=str, default="flickr30k_eval_results.xlsx",
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

    # Max vs average references
    parser.add_argument("--use-max-ref", action="store_true", default=False,
                        help="If set, compute max per-sample score over 5 references (default: average)")

    args = parser.parse_args()

    # Load results
    with open(args.inference_results, "r") as f:
        results = json.load(f)
    predictions = {int(k): v for k, v in results.get("predictions", {}).items()}
    ground_truths = {int(k): v for k, v in results.get("ground_truths", {}).items()}
    inference_times = {int(k): v for k, v in results.get("inference_times", {}).items()}
    vram_usage = {int(k): v for k, v in results.get("vram_usage", {}).items()}
    sample_indices = results.get("sample_indices", [])
    prompt = results.get("prompt", "")

    # Prepare lists
    pred_list = [predictions[idx] for idx in sample_indices]
    gt_list = [ground_truths[idx] for idx in sample_indices]
    average_over_refs = not args.use_max_ref

    # Compute metrics
    print("\n🔄 Computing CIDEr, SPICE, Cosine similarity scores...")
    corpus_cider, per_sample_cider = compute_cider_scores(pred_list, gt_list, average_over_refs)
    corpus_spice, per_sample_spice = compute_spice_scores(pred_list, gt_list, average_over_refs)
    corpus_cosine, per_sample_cosine = compute_cosine_scores(pred_list, gt_list, average_over_refs)

    print(f"✅ Evaluation complete!")
    print(f"   - CIDEr (corpus): {corpus_cider:.4f}")
    print(f"   - SPICE (corpus): {corpus_spice:.4f}")
    print(f"   - Cosine (corpus): {corpus_cosine:.4f}")

    # Load dataset for images
    test_dataset = datasets.load_from_disk(args.test_dataset)

    # Log to Excel
    log_metrics_to_excel(
        sample_indices, predictions, ground_truths,
        per_sample_cider, per_sample_spice, per_sample_cosine,
        inference_times, vram_usage, test_dataset, prompt,
        args.model_name, args.output_excel
    )

    # Log to WandB
    if args.use_wandb:
        log_metrics_to_wandb(
            sample_indices, predictions, ground_truths,
            per_sample_cider, per_sample_spice, per_sample_cosine,
            inference_times, vram_usage, test_dataset, prompt,
            args.model_name, args.wandb_project, args.wandb_run_name
        )

    print("\n✅ Evaluation complete!")
