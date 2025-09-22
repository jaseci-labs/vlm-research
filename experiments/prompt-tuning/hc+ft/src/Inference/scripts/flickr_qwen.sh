#!/bin/bash

# --- variables: change names if needed ---
ZIP="finetuned_model.zip"
DEST="unsloth_finetune"
DATASET_FOLDER="/workspace/Captioned_Data/filtered_images"
RUN_SCRIPT="Inference.py"   
MODEL_DIR="unsloth_finetune"
WANDB_PROJECT="flickr-eval"
MODEL_NAME="unsloth/Qwen2-VL-7B-Instruct"

# Add sample indices to exclude from training
EXCLUDE_INDICES=(2500 2501 2502 2503 2504 2505 2506 2507 2508 2509)

# --- list of prompts ---
PROMPTS=("Describe the image" "What is happening in the image?" "List objects in the image" "Summarize the scene")

# --- prepare target dir ---
mkdir -p "$DEST"

# --- unzip (overwrite if already present) ---
echo "➡️ Unzipping $ZIP -> $DEST"
unzip -o "$ZIP" -d "$DEST" || { echo "❌ unzip failed"; exit 1; }

# --- show top-level contents for sanity ---
echo "🔎 Top-level contents of $DEST:"
ls -la "$DEST" | sed -n '1,200p'

# --- loop over prompts ---
for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[i]}"
    NUM=$(printf "%02d" $((i+1)))   # 01, 02, 03, 04

    OUTPUT_XLS="Flickr_prompt${NUM}.xlsx"

    echo "🚀 Running evaluation for prompt: \"$PROMPT\""
    
    python flickr_ft.py \
        --model_name "$MODEL_NAME" \
        --save_dir "$ZIP" \
        --prompt "$PROMPT" \
        --exclude "${EXCLUDE_INDICES[@]}"

    if [ -f "$RUN_SCRIPT" ]; then
        python "$RUN_SCRIPT" \
            --prompt "$PROMPT" \
            --model-name "$MODEL_NAME" \
            --dataset-folder "$DATASET_FOLDER" \
            --wandb-project "$WANDB_PROJECT" \
            --output-excel "$OUTPUT_XLS"
        echo "✅ Done. Excel saved at: $OUTPUT_XLS"
    else
        echo "❗ $RUN_SCRIPT not found in cwd. If you don't have it, run your own eval script and pass --model-name or --model-path as $MODEL_ROOT"
        exit 1
    fi
done

echo "🏁 All prompts processed."
