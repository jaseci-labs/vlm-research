#!/bin/bash
set -e  # Stop the script on any error

chmod +x unslothinstall.sh
./unslothinstall.sh

source unsloth_env/bin/activate

unzip flickr_subset.zip
# --- variables: change names if needed ---
SAVE_DIR="unsloth_finetune"
DATASET_FOLDER="workspace/train"
RUN_SCRIPT="Inference.py"   
WANDB_PROJECT="flickr-eval"
MODEL_NAME="unsloth/Qwen2-VL-7B-Instruct"
SAMPLE_FOLDER="flickr_sample_hf/train"
USE_HF_DOWNLOAD=true 

HF_TOKEN=""  # add your huggingface token here
REPO_ID=""  # add your huggingface repo id here

# Add sample indices to exclude from training
EXCLUDE_INDICES=(2500 2501 2502 2503 2504 2505 2506 2507 2508 2509)

# --- list of prompts ---
PROMPTS=(“an image of…”,
"You are an observer describing the scene. Explain what is happening, focusing on the actions of people, animals, or objects as if you were narrating it for someone else.",
"Imagine you are a guide or commentator. Describe how a <text_1> is interacting with a <text_2> in the environment, providing context and details.",
"Describe using format - Subject: ___; Activity: ___; Environment: ___; Additional Notes: ___;"
)

# --- prepare target dir ---
mkdir -p "$SAVE_DIR"

# --- loop over prompts ---
for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[i]}"
    NUM=$(printf "%02d" $((i+1)))   # 01, 02, 03, 04

    OUTPUT_XLS="Flickr_prompt${NUM}.xlsx"
    RUN_REPO_ID="${REPO_ID}-prompt${NUM}"
    echo "🚀 Running evaluation for prompt: \"$PROMPT\""
    
    if [ "$USE_HF_DOWNLOAD" = true ]; then
        echo "➡️ Using model adapters from Hugging Face repo_id: $REPO_ID"
        MODEL_DIR="RUN_REPO_ID"
        echo $MODEL_DIR
    else
        echo "➡️ Running flickr_ft.py script"
        python flickr_ft.py \
            --model_name "$MODEL_NAME" \
            --save_dir "$SAVE_DIR" \
            --prompt "$PROMPT" \
            --hf_token "$HF_TOKEN" \
            --exclude "${EXCLUDE_INDICES[@]}" \
            --repo_id "$RUN_REPO_ID"
    fi
    echo "Running inference"
    if [ -f "$RUN_SCRIPT" ]; then
        python "$RUN_SCRIPT" \
            --prompt "$PROMPT" \
            --model-name "$MODEL_NAME" \
            --dataset-folder "$DATASET_FOLDER" \
            --wandb-project "$WANDB_PROJECT" \
            --output-excel "$OUTPUT_XLS" \
            --model-dir "$MODEL_DIR" \
            --load-from-hf #remove this flag if not loading from HF
        echo "✅ Done. Excel saved at: $OUTPUT_XLS"
    else
        echo "❗ $RUN_SCRIPT not found in cwd. If you don't have it, run your own eval script and pass --model-name or --model-path as $MODEL_ROOT"
        exit 1
    fi
done

echo "🏁 All prompts processed."
