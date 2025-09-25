import os
import argparse
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as ExcelImage
from PIL import Image as PILImage
from io import BytesIO
import pandas as pd
from datasets import load_dataset, load_from_disk
import json
#Initialize wandb
import wandb
from utils import evaluate_batch

import argparse
import io
import os
from typing import List, Any

import pandas as pd
from PIL import Image


import pandas as pd
from typing import List, Any
from io import BytesIO
from PIL import Image as PILImage
import wandb

def log_metrics_to_excel(
    samples: List[int],
    model_name: str,
    results: List[Any],
    inference_times: List[float],
    vram_usage: List[float],
    cosine_scores: List[float],
    spice_scores: List[float],
    cider_scores: List[float],
    flickr_subset,
    output_excel_path: str = "Flickr_pixtral.xlsx",
    prompts: str = None,
    wandb_project: str = "flickr-eval"
):
    rows = []
    pil_images = []  # keep images for wandb and excel insertion
    n = len(samples)
    for i, s in enumerate(samples):
        # Handle both dict and list formats
        pred = results[s] if isinstance(results, dict) else (results[i] if i < len(results) else None)
        time_taken = inference_times[s] if isinstance(inference_times, dict) else (inference_times[i] if i < len(inference_times) else None)
        vram = vram_usage[s] if isinstance(vram_usage, dict) else (vram_usage[i] if i < len(vram_usage) else None)
        cos = cosine_scores[s] if isinstance(cosine_scores, dict) else (cosine_scores[i] if i < len(cosine_scores) else None)
        spice = spice_scores[s] if isinstance(spice_scores, dict) else (spice_scores[i] if i < len(spice_scores) else None)
        cider = cider_scores[s] if isinstance(cider_scores, dict) else (cider_scores[i] if i < len(cider_scores) else None)

        # sample lookup
        sample_item = flickr_subset[s]
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
            "cosine_score": cos,
            "spice_score": spice,
            "cider_score": cider,
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
    table_cols = ["sample_index", "image", "prediction", "cosine_score", "spice_score", "cider_score", "vram_usage", "inference_time_s", "prompt"]
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
            row["cosine_score"],
            row["spice_score"],
            row["cider_score"],
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
    parser.add_argument("--dataset-folder", type=str, default="/workspace/filtered_dataset", help="Fallback image folder (if dataset items are paths)")
    parser.add_argument("--wandb-project", type=str, default="flickr-eval", help="WandB project name")
    parser.add_argument("--output-excel", type=str, default="Flickr_pixtral.xlsx", help="Output Excel file path")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory of the fine-tuned model (if any)")
    parser.add_argument("--load-from-hf", action="store_true", help="Whether to load the model from Hugging Face Hub")
    args = parser.parse_args()

    prompt = args.prompt
    model_name = args.model_name
    dataset_folder = args.dataset_folder
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

    print("🔄 Loading Car Damage subset dataset...")
    try:
        flickr_subset = load_from_disk(dataset_folder)
        print("✅ Dataset loaded. Number of samples:", len(flickr_subset))
    except Exception as e:
        print(f"[warning] Could not load dataset via load_from_disk({dataset_folder}): {e}")
        # fallback: try to treat dataset_folder as a directory of images
        flickr_subset = []
        print("⚠️ flickr_subset is empty; images will be looked up from --img-folder by index when possible")

    print(f"🚀 Running evaluation batch with model {model_name}...")
    results, cosine_scores, spice_scores, cider_scores, inference_times, vram_usage = evaluate_batch(
            prompt,
            flickr_subset,
            samples,
            multiple_refs,
            MODEL_DIR=model_dir,
            LOAD_FROM_HF=args.load_from_hf
        )


    print("✅ Evaluation complete!")
    print("📊 Results summary:")
    print("Cosine scores:", cosine_scores)
    print("SPICE scores:", spice_scores)
    print("CIDER scores:", cider_scores)
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
        cosine_scores,
        spice_scores,
        cider_scores,
        flickr_subset,
        output_excel_path=excel_path,
        prompts=prompt,
        wandb_project=wandb_project
    )
    print(f"✅ Metrics logged to {excel_path} and to wandb (if configured).")
