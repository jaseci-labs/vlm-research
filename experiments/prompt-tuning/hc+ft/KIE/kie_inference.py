import os
import argparse
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as ExcelImage
from io import BytesIO
import pandas as pd
from datasets import load_dataset, load_from_disk
import json 
from utils import evaluate_batch

import argparse
from typing import List, Any
from PIL import Image as PILImage
import wandb

def log_metrics_to_excel(
    samples: List[int],
    model_name: str,
    results: List[Any],
    inference_times: List[float],
    vram_usage: List[float],
    kie_scores: List[float],
    test_subset,
    output_excel_path: str = "KIE_pixtral.xlsx",
    prompts: str = None,
    wandb_project: str = "kie-eval"
):
    rows = []
    pil_images = []  # keep images for wandb and excel insertion
    n = len(samples)
    for i, s in enumerate(samples):
        pred = results.get(s, None)
        time_taken = inference_times.get(s, None)
        vram       = vram_usage.get(s, None)
        kie = kie_scores[i] if i < len(kie_scores) else None

        # sample lookup
        sample_item = test_subset[s]
        pil_img = sample_item['image']
        if not isinstance(pil_img, PILImage.Image):
            pil_img = PILImage.fromarray(pil_img)
        pil_images.append(pil_img)  # ✅ now stored for wandb & excel

        raw_caption = sample_item['caption']
        caption_text = "\n".join(str(c) for c in raw_caption) if isinstance(raw_caption, list) else str(raw_caption)
        
        row = {
            "sample_index": s,
            "prompt": prompts,
            "original_caption": caption_text,
            "prediction": pred,
            "kie_score": kie,
            "vram_usage": vram,
            "inference_time_s": time_taken,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # --- write DataFrame to Excel ---
    writer = pd.ExcelWriter(output_excel_path, engine="xlsxwriter")
    sheet_name = "evaluation"
    df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=1, index=False)

    workbook  = writer.book
    worksheet = writer.sheets[sheet_name]

    # insert images into column A
    for row_idx, pil_img in enumerate(pil_images, start=1):  # row 1 = header row
        img_stream = BytesIO()
        pil_img.thumbnail((128, 128)) 
        pil_img.save(img_stream, format="PNG")
        img_stream.seek(0)
        worksheet.insert_image(row_idx, 0, f"image_{row_idx}.png", {"image_data": img_stream})

    writer.close()  # ✅ ensures images + data are written

    # --- log to wandb ---
    try:
        run = wandb.init(project=wandb_project, reinit=True)
    except Exception as e:
        print(f"[warning] wandb.init failed: {e}. Skipping wandb logging.")
        return df, None
    table_cols = ["sample_index", "image", "prediction", "kie_score", "vram_usage", "inference_time_s", "prompt"]
    wandb_table = wandb.Table(columns=table_cols)
    for i, row in enumerate(rows):
        pil_img = pil_images[i]
        wb_image = None
        if pil_img is not None:
            try:
                wb_image = wandb.Image(pil_img, caption=f"sample_{row['sample_index']}")
            except Exception:
                wb_image = None
        wandb_table.add_data(
            row["sample_index"],
            wb_image,
            row["prediction"],
            row["kie_score"],
            row["vram_usage"],
            row["inference_time_s"],
            row["prompt"],
        )
    wandb.log({"evaluation_table": wandb_table})
    try:
        wandb.finish()
    except Exception:
        pass
    print(f"✅ Excel written to {output_excel_path} and table logged to wandb project '{wandb_project}'.")
    return df,wandb_table


if __name__ == "__main__":
    import argparse
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run inference on a vision-language model")
    parser.add_argument("--prompt", type=str, default="Explain the image content step by step.", help="Prompt for the model")
    parser.add_argument("--model-name", type=str, default="Pixtral-12B", help="Model name to evaluate (e.g. Pixtral-12B)")
    parser.add_argument("--subset-repo", type=str, default="/workspace/filtered_dataset", help="Fallback image folder (if dataset items are paths)")
    parser.add_argument("--wandb-project", type=str, default="flickr-eval", help="WandB project name")
    parser.add_argument("--output-excel", type=str, default="Flickr_pixtral.xlsx", help="Output Excel file path")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory of the fine-tuned model (if any)")
    parser.add_argument("--load-from-hf", action="store_true", help="Whether to load the model from Hugging Face Hub")
    args = parser.parse_args()

    prompt = args.prompt
    model_name = args.model_name
    subset_repo = args.subset_repo
    wandb_project = args.wandb_project
    excel_path = args.output_excel
    model_dir= args.model_dir

    wandb.init(
        project="Prompting-Experiments",
        name=model_name,
        group="prompting-experiments",
        tags=[model_name, "prompting", "vision", "finetune"],
        notes=f"Testing OCR accuracy with {model_name} + fine-tuned prompts.",
        config={
            "learning_rate": 2e-4,
            "batch_size": 2,
            "model": model_name,
            "prompting": "handcrafted+fine-tune",
        },
    )
    samples = [0,1,2,3,4,5,6,7,8,9]
    multiple_refs = True

    
    print("🔄 Loading Flickr subset dataset...")
    try:
        test_subset = load_dataset(subset_repo)
        print("✅ Dataset loaded. Number of samples:", len(test_subset))
    except Exception as e:
        print(f"[warning] Could not load dataset via load_from_disk({subset_repo}): {e}")
        # fallback: try to treat dataset_folder as a directory of images
        test_subset = []
        print("⚠️ test_subset is empty; images will be looked up from --img-folder by index when possible")

    print(f"🚀 Running evaluation batch with model {model_name}...")
    # NOTE: ensure your evaluate_batch signature accepts model_name parameter, or adjust accordingly.
    # I pass model_name as a keyword argument — if evaluate_batch doesn't accept it, change evaluate_batch to accept it.
    results, kie_scores, inference_times, vram_usage = evaluate_batch(
            prompt,
            test_subset,
            samples,
            multiple_refs,
            MODEL_DIR=model_dir,
            LOAD_FROM_HF=args.load_from_hf
        )

    print("✅ Evaluation complete!")
    print("📊 Results summary:")
    print("KIE scores:", kie_scores)
    print("Inference times:", inference_times)
    print("VRAM usage:", vram_usage)

    print("💾 Logging metrics to Excel & wandb...")
    # table, wandbtable = log_metrics_to_excel(
    table = log_metrics_to_excel(
        samples,
        model_name,
        results,
        inference_times,
        vram_usage,
        kie_scores,
        test_subset,
        output_excel_path=excel_path,
        prompts=prompt,
        wandb_project=wandb_project
    )
    print(f"✅ Metrics logged to {excel_path} and to wandb (if configured).")
