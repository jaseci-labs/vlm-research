!/bin/bash
set -e  # Stop the script on any error

#if you get this error --> bash: ./flickr_qwen.sh: /bin/bash^M: bad interpreter: No such file or directory
#run
#apt update && apt install dos2unix && dos2unix unslothinstall.sh && dos2unix cardd_qwen.sh
#chmod +x cardd_qwen.sh && ./cardd_qwen.sh

chmod +x unslothinstall.sh
./unslothinstall.sh

source unsloth_env/bin/activate

# --- variables: change names if needed ---
SAVE_DIR="unsloth_finetune"
RUN_SCRIPT="Inference.py"   
WANDB_PROJECT="cardd-eval"
MODEL_NAME="unsloth/Qwen2-VL-7B-Instruct"
USE_HF_DOWNLOAD=false 
DATASET_REPO="RR32444/cardd_dataset"
SAMPLE_REPO="RR32444/cardd_subset"
MODEL_DIR="$SAVE_DIR"

HF_TOKEN=""  # add your huggingface token here
REPO_ID=""  # add your huggingface repo id here


# --- list of prompts ---
PROMPTS=("an image of...",
        "Describe &&damage 12 sedan drive’ this !!image.",
            "Explain the visible damage to this vehicle. Question: What areas are affected and how severe is the damage? Answer:",
            "You are an insurance claims assessor. Provide a detailed description of the car’s condition.",
            "This \<part\_1> of the car has \<damage_type\_1> . The severity appears to be \<severity\_1>. Additional notes: \<text\_1>.",
            "Describe the damage in the following format – Damage Type: \_\_\_; Affected Part: \_\_\_; Severity: \_\_\_; Notes: \_\_\_"
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
        echo "➡️ Downloading model from Hugging Face repo_id: $REPO_ID"
        MODEL_DIR="$RUN_REPO_ID"
        echo $MODEL_DIR
    else
        MODEL_DIR="$SAVE_DIR"
        echo "➡️ Running cardd_ft.py script"
        python cardd_ft.py \
            --model_name "$MODEL_NAME" \
            --datast_repo "$DATASET_REPO" \
            --save_dir "$SAVE_DIR" \
            --prompt "$PROMPT" \
            --hf_token "$HF_TOKEN" \
            --repo_id "$RUN_REPO_ID"
    fi

    echo "Running inference"

    if [ -f "$RUN_SCRIPT" ]; then
        # Build the command
        CMD="python \"$RUN_SCRIPT\" \
            --prompt \"$PROMPT\" \
            --model-name \"$MODEL_NAME\" \
            --pickle-path \"cardd-df.p\" \
            --dataset-repo \"$SAMPLE_REPO\" \
            --wandb-project \"$WANDB_PROJECT\" \
            --output-excel \"$OUTPUT_XLS\" \
            --model-dir \"$MODEL_DIR\""
        # Conditionally add HF flag
        if [ "$USE_HF_DOWNLOAD" = true ]; then
            CMD="$CMD --load-from-hf"
        fi

        # Run it
        eval $CMD
        echo "✅ Done. Excel saved at: $OUTPUT_XLS"

    else
        echo "❗ $RUN_SCRIPT not found in cwd. If you don't have it, run your own eval script and pass --model-name or --model-path as $MODEL_ROOT"
        exit 1
    fi
   
    
done

echo "🏁 All prompts processed."
