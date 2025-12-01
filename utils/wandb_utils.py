#!/usr/bin/env python3
"""
Weights & Biases logging utilities for VLM research experiments.
"""

import json
import wandb
from typing import Dict, List, Any, Optional
import pandas as pd
from io import BytesIO
from PIL import Image as PILImage


# Default WandB project name for all experiments
DEFAULT_WANDB_PROJECT = "vlm-research"


def init_wandb(
    project: str = DEFAULT_WANDB_PROJECT,
    name: Optional[str] = None,
    config: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    reinit: bool = False
) -> Any:
    """
    Initialize a WandB run with consistent project naming.

    Args:
        project: WandB project name (default: vlm-research)
        name: Run name (optional)
        config: Configuration dict to log
        tags: List of tags for the run
        group: Group name for organizing related runs
        reinit: Whether to reinitialize if already initialized

    Returns:
        WandB run object

    Example:
        >>> run = init_wandb(name="kie_baseline_gemma3", tags=["kie", "baseline"])
    """
    try:

        run = wandb.init(
            project=project,
            name=name,
            config=config or {},
            tags=tags or [],
            group=group,
            reinit=reinit
        )
        print(f"✅ WandB initialized: {project}/{name or 'unnamed'}")
        return run

    except ImportError:
        print("⚠️  WandB not installed. Install with: pip install wandb")
        return None
    except Exception as e:
        print(f"⚠️  WandB initialization failed: {e}")
        print("   Possible solutions:")
        print("   1. Run: wandb login")
        print("   2. Set: export WANDB_API_KEY=your_key")
        return None


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """
    Log metrics to WandB.

    Args:
        metrics: Dictionary of metric name -> value
        step: Optional step number
    """
    try:
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception as e:
        print(f"⚠️  Failed to log to WandB: {e}")


def log_metrics_to_excel(
    sample_indices: List[int],
    predictions: Dict[int, str],
    ground_truths: Dict[int, str],
    scores: List[float],
    inference_times: Dict[int, float],
    vram_usage: Dict[int, float],
    test_dataset: Any,
    prompt: str,
    model_name: str,
    output_excel_path: str = "results.xlsx",
    score_name: str = "kie_score"
) -> pd.DataFrame:
    """
    Log evaluation metrics to an Excel file with images.

    Args:
        sample_indices: List of sample indices
        predictions: Dictionary of predictions (idx -> pred)
        ground_truths: Dictionary of ground truths (idx -> gt)
        scores: List of scores per sample
        inference_times: Dictionary of inference times
        vram_usage: Dictionary of VRAM usage
        test_dataset: Test dataset with images
        prompt: Prompt used for inference
        model_name: Model name for logging
        output_excel_path: Path to save Excel file
        score_name: Name of the score column

    Returns:
        DataFrame with results
    """
    rows = []
    pil_images = []

    for i, idx in enumerate(sample_indices):
        pred = predictions.get(idx, "")
        gt = ground_truths.get(idx, "")
        time_taken = inference_times.get(idx, 0.0)
        vram = vram_usage.get(idx, 0.0)
        score = scores[i] if i < len(scores) else 0.0

        # Get image from dataset
        sample_item = test_dataset[idx]
        pil_img = sample_item['image']

        # Handle different image types
        if isinstance(pil_img, bytes):
            pil_img = PILImage.open(BytesIO(pil_img))
        elif not isinstance(pil_img, PILImage.Image):
            pil_img = PILImage.fromarray(pil_img)

        pil_images.append(pil_img)

        # Format ground truth
        if isinstance(gt, dict):
            gt_text = json.dumps(gt, indent=2)
        elif isinstance(gt, list):
            gt_text = "\n".join(str(c) for c in gt)
        else:
            gt_text = str(gt)

        row = {
            "sample_index": idx,
            "prompt": prompt,
            "model": model_name,
            "ground_truth": gt_text,
            "prediction": pred,
            score_name: score,
            "inference_time_s": time_taken,
            "vram_usage_mb": vram,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Write DataFrame to Excel with images
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
    worksheet.set_column(0, 0, 20)   # Image column
    worksheet.set_column(1, 1, 12)   # sample_index
    worksheet.set_column(2, 2, 50)   # prompt
    worksheet.set_column(3, 3, 20)   # model
    worksheet.set_column(4, 4, 40)   # ground_truth
    worksheet.set_column(5, 5, 40)   # prediction
    worksheet.set_column(6, 6, 12)   # score
    worksheet.set_column(7, 7, 15)   # inference_time_s
    worksheet.set_column(8, 8, 15)   # vram_usage_mb

    writer.close()

    print(f"✅ Excel written to: {output_excel_path}")

    # Print summary stats
    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_time = sum(inference_times.values()) / len(inference_times) if inference_times else 0.0
    avg_vram = sum(vram_usage.values()) / len(vram_usage) if vram_usage else 0.0

    print(f"\n📊 Summary Statistics:")
    print(f"   - Average {score_name}: {avg_score:.4f}")
    print(f"   - Average Inference Time: {avg_time:.3f}s")
    print(f"   - Average VRAM Usage: {avg_vram:.2f} MB")

    return df


def log_metrics_to_wandb(
    sample_indices: List[int],
    predictions: Dict[int, str],
    ground_truths: Dict[int, str],
    scores: List[float],
    inference_times: Dict[int, float],
    vram_usage: Dict[int, float],
    test_dataset: Any,
    prompt: str,
    model_name: str,
    project: str = DEFAULT_WANDB_PROJECT,
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    score_name: str = "kie_score"
):
    """
    Log evaluation metrics to WandB with images and table.

    Args:
        sample_indices: List of sample indices
        predictions: Dictionary of predictions
        ground_truths: Dictionary of ground truths
        scores: List of scores per sample
        inference_times: Dictionary of inference times
        vram_usage: Dictionary of VRAM usage
        test_dataset: Test dataset with images
        prompt: Prompt used
        model_name: Model name
        project: WandB project name
        run_name: WandB run name (optional)
        tags: List of tags
        score_name: Name of the score metric
    """

    try:
        run = wandb.init(
            project=project,
            name=run_name,
            tags=tags or [],
            reinit=True
        )

        # Log summary metrics
        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_time = sum(inference_times.values()) / len(inference_times) if inference_times else 0.0
        avg_vram = sum(vram_usage.values()) / len(vram_usage) if vram_usage else 0.0

        wandb.log({
            f"avg_{score_name}": avg_score,
            "avg_inference_time_s": avg_time,
            "avg_vram_usage_mb": avg_vram,
        })

        # Create evaluation table
        table_cols = [
            "sample_index", "image", "prediction", "ground_truth",
            score_name, "inference_time_s", "vram_usage_mb", "prompt"
        ]
        wandb_table = wandb.Table(columns=table_cols)

        for i, idx in enumerate(sample_indices):
            pred = predictions.get(idx, "")
            gt = ground_truths.get(idx, "")
            time_taken = inference_times.get(idx, 0.0)
            vram = vram_usage.get(idx, 0.0)
            score = scores[i] if i < len(scores) else 0.0

            # Get image
            sample_item = test_dataset[idx]
            pil_img = sample_item['image']

            if isinstance(pil_img, bytes):
                pil_img = PILImage.open(BytesIO(pil_img))
            elif not isinstance(pil_img, PILImage.Image):
                pil_img = PILImage.fromarray(pil_img)

            try:
                wb_image = wandb.Image(pil_img, caption=f"sample_{idx}")
            except Exception:
                wb_image = None

            # Format ground truth
            if isinstance(gt, dict):
                gt_str = json.dumps(gt, indent=2)
            else:
                gt_str = str(gt)

            wandb_table.add_data(idx, wb_image, pred, gt_str, score, time_taken, vram, prompt)

        wandb.log({"evaluation_table": wandb_table})
        wandb.finish()

        print(f"✅ Metrics logged to WandB project: {project}")

    except Exception as e:
        print(f"⚠️  Failed to log to WandB: {e}")


def finish_wandb():
    """Finish the current WandB run."""
    try:
        if wandb.run is not None:
            wandb.finish()
            print("✅ WandB run completed")
    except Exception as e:
        print(f"⚠️  Warning: Failed to finish WandB run: {e}")
