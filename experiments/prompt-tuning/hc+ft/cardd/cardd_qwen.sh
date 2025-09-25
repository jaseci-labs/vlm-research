#!/bin/bash
set -e  # Stop the script on any error
cd "$(dirname "$0")"  # Set working directory to the script's location

chmod +x unslothinstall.sh
./unslothinstall.sh

source unsloth_env/bin/activate

unzip cardd_subset.zip
unzip cardd_dataset.zip

# --- variables: change names if needed ---
SAVE_DIR="unsloth_finetune"
DATASET_FOLDER="/workspace/VLM-Playground/experiments/ft+hc-prompting/cardd/cardd_dataset/kaggle/working/cardd_data_hf/train"
RUN_SCRIPT="Inference.py"
WANDB_PROJECT="cardd-eval"
MODEL_NAME="unsloth/Qwen2-VL-7B-Instruct"
SAMPLE_FOLDER="train"
USE_HF_DOWNLOAD=false 

HF_TOKEN=""  # add your huggingface token here
REPO_ID=""  # add your huggingface repo id here


# --- list of prompts ---
PROMPTS=("an image of..."
            # ,"Explain the visible damage to this vehicle. Question: What areas are affected and how severe is the damage? Answer:",
            # "You are an insurance claims assessor. Provide a detailed description of the car’s condition.",
            # "This \<part\_1> of the car has \<damage_type\_1> . The severity appears to be \<severity\_1>. Additional notes: \<text\_1>.",
            # "Describe the damage in the following format – Damage Type: \_\_\_; Affected Part: \_\_\_; Severity: \_\_\_; Notes: \_\_\_"
)

mkdir -p "$SAVE_DIR"

# --- loop over prompts ---
for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[i]}"
    NUM=$(printf "%02d" $((i+1)))   # 01, 02, 03, 04

    OUTPUT_XLS="cardd_prompt${NUM}.xlsx"
    RUN_REPO_ID="${REPO_ID}-prompt${NUM}"
    echo "🚀 Running evaluation for prompt: \"$PROMPT\""

    if [ "$USE_HF_DOWNLOAD" = true ]; then
        echo "➡️ Loading model from Hugging Face repo: $RUN_REPO_ID"
        MODEL_DIR="$RUN_REPO_ID"
        echo "🔍 Model directory set to: $MODEL_DIR"
    else
        echo "➡️ Fine-tuning model locally first"
        MODEL_DIR="$SAVE_DIR"
        echo "🔍 Model directory set to: $MODEL_DIR"
        python cardd_ft.py \
            --model_name "$MODEL_NAME" \
            --dataset_folder "$DATASET_FOLDER" \
            --save_dir "$SAVE_DIR" \
            --prompt "$PROMPT" \
            --hf_token "$HF_TOKEN" \
            --repo_id "$RUN_REPO_ID"
    fi

    echo "Running inference"
    if [ -f "$RUN_SCRIPT" ]; then
        if [ "$USE_HF_DOWNLOAD" = true ]; then
            python "$RUN_SCRIPT" \
                --prompt "$PROMPT" \
                --model-name "$MODEL_NAME" \
                --dataset-folder "$SAMPLE_FOLDER" \
                --wandb-project "$WANDB_PROJECT" \
                --output-excel "$OUTPUT_XLS" \
                --model-dir "$MODEL_DIR" \
                --load-from-hf
        else
            python "$RUN_SCRIPT" \
                --prompt "$PROMPT" \
                --model-name "$MODEL_NAME" \
                --dataset-folder "$SAMPLE_FOLDER" \
                --wandb-project "$WANDB_PROJECT" \
                --output-excel "$OUTPUT_XLS" \
                --model-dir "$MODEL_DIR"
        fi
        echo "✅ Done. Excel saved at: $OUTPUT_XLS"
    else
        echo "❗ $RUN_SCRIPT not found in cwd. If you don't have it, run your own eval script and pass --model-name or --model-path as $MODEL_ROOT"
        exit 1
    fi
done

echo "🏁 All prompts processed."
