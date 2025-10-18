#!/bin/bash
set -e  # Stop the script on any error

chmod +x unslothinstall.sh
./unslothinstall.sh

source unsloth_env/bin/activate

# --- variables: change names if needed ---
SAVE_DIR="unsloth_finetune"
RUN_SCRIPT="Inference.py"   
WANDB_PROJECT="cardd-eval"
MODEL_NAME="unsloth/Qwen2-VL-7B-Instruct"
USE_HF_DOWNLOAD=false 
#DATASET_REPO="RR32444/cardd_dataset"
SAMPLE_REPO="RR32444/kie_subset"
MODEL_DIR="$SAVE_DIR"

HF_TOKEN=""  # add your huggingface token here
REPO_ID=""  # add your huggingface repo id here


# --- list of prompts ---
PROMPTS=( """You are a highly accurate document understanding agent designed to extract structured information from scanned receipts, invoices, and sales slips.

Your goal is to extract a fixed set of predefined fields from a given document image and return them as a single well-formed JSON object.

Follow these rules carefully:
1. Match Labels and Synonyms: Use exact field labels or common variations (e.g., "Tax ID", "GST No.", "TIN").
2. Position Awareness: Use the layout of the document to infer missing labels (e.g., phone number near store name).
3. Text Cleanup: Remove OCR noise, headers, and irrelevant content.
4. Currency Handling: Preserve currency symbols and decimal formatting in monetary values.
5. Missing or Unreadable Fields: If a field is not present or unreadable, return its value as "".
6. Field Consistency: Always return the same 8 fields, in the exact order shown below.

Output format must strictly match this schema:
{
  "date": "DD/MM/YYYY or similar format",
  "doc_no_receipt_no": "...",
  "seller_name": "...",
  "seller_address": "...",
  "seller_phone": "...",
  "seller_gst_id": "...",
  "total_tax": "...",
  "total_amount": "..."
}"""
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
            --save_dir "$SAVE_DIR" \
            --prompt "$PROMPT" \
            --hf_token "$HF_TOKEN" \
            --repo_id "$RUN_REPO_ID"
    fi

    echo "Running inference"
    if [ -f "$RUN_SCRIPT" ]; then
        CMD="python \"$RUN_SCRIPT\" \
            --prompt \"$PROMPT\" \
            --model-name \"$MODEL_NAME\" \
            --subset-repo \"$SAMPLE_REPO\" \
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
