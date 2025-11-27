#!/bin/bash
# ============================================================================
# This script fine-tunes the Gemma-3-12B-IT model on KIE dataset.
# It loops over all prompts in prompts.yml and creates a fine-tuned model
# for each prompt. Optionally uploads models to HuggingFace.
# ============================================================================

set -e  # Stop on any error

# Model configuration
BASE_MODEL="unsloth/gemma-3-12b-it"
TRAIN_DATASET="./kie_splits/train"
EVAL_DATASET="./kie_splits/eval"
SAVE_DIR_BASE="./kie_finetuned"

# HuggingFace upload (set to true to enable)
UPLOAD_TO_HF=false
HF_TOKEN=""  # Add your HuggingFace token here
HF_REPO_BASE=""  # Base repo ID (e.g., "Gayanukaa/vlm-finetunes")
                 # Models will be saved as: {HF_REPO_BASE}-{prompt_key}
                 # Examples: Gayanukaa/vlm-finetunes-baseline
                 #           Gayanukaa/vlm-finetunes-masked

# WandB logging (set to true to enable)
USE_WANDB=false
WANDB_PROJECT="kie-finetuning"
WANDB_RUN_NAME_PREFIX="kie_ft"  # Will be appended with prompt key and timestamp

# LoRA configuration
LORA_R=8
LORA_ALPHA=8
LORA_DROPOUT=0.01

# Training hyperparameters
LEARNING_RATE=2e-4
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
WARMUP_RATIO=0.13
MAX_STEPS=75
FP16=true
OPTIMIZER="adamw_8bit"
LR_SCHEDULER="cosine"
WEIGHT_DECAY=0.01
MAX_SEQ_LENGTH=2048
SEED=3407

# Prompts file
PROMPTS_FILE="prompts.yml"

# HELPER FUNCTIONS

# Function to extract prompts from YAML file
extract_prompts() {
    python3 << 'EOF'
import yaml
import sys

try:
    with open("prompts.yml", "r") as f:
        prompts_data = yaml.safe_load(f)

    prompts = []
    for key, value in prompts_data.items():
        if isinstance(value, dict) and "text" in value:
            prompt_text = value["text"].strip()
            prompts.append((key, prompt_text))

    # Output as JSON for easier parsing in bash
    import json
    print(json.dumps(prompts))
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

# MAIN EXECUTION

echo "============================================================================"
echo "🚀 KIE Fine-tuning Script"
echo "============================================================================"
echo ""
echo "Model Configuration:"
echo "  - Base Model: $BASE_MODEL"
echo "  - Training Dataset: $TRAIN_DATASET"
echo "  - Evaluation Dataset: $EVAL_DATASET"
echo "  - Save Directory: $SAVE_DIR_BASE"
echo ""
echo "LoRA Configuration:"
echo "  - Rank: $LORA_R"
echo "  - Alpha: $LORA_ALPHA"
echo "  - Dropout: $LORA_DROPOUT"
echo ""
echo "Training Hyperparameters:"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  - Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  - Warmup Ratio: $WARMUP_RATIO"
echo "  - Max Steps: $MAX_STEPS"
echo "  - Optimizer: $OPTIMIZER"
echo "  - LR Scheduler: $LR_SCHEDULER"
echo "  - Weight Decay: $WEIGHT_DECAY"
echo "  - Precision: FP16=$FP16"
echo ""
echo "HuggingFace Upload: $UPLOAD_TO_HF"
if [ "$UPLOAD_TO_HF" = true ]; then
    echo "  - Base Repository: $HF_REPO_BASE"
fi
echo ""
echo "WandB Logging: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
    echo "  - Project: $WANDB_PROJECT"
    echo "  - Run Name Prefix: $WANDB_RUN_NAME_PREFIX"
fi
echo ""
echo "============================================================================"
echo ""

# Check if datasets exist
if [ ! -d "$TRAIN_DATASET" ]; then
    echo "❌ Error: Training dataset not found at $TRAIN_DATASET"
    echo "Please run './0_prepare_data.sh' first to create the dataset splits."
    exit 1
fi

if [ ! -d "$EVAL_DATASET" ]; then
    echo "❌ Error: Evaluation dataset not found at $EVAL_DATASET"
    echo "Please run './0_prepare_data.sh' first to create the dataset splits."
    exit 1
fi

# Check if prompts file exists
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "❌ Error: Prompts file not found at $PROMPTS_FILE"
    exit 1
fi

# Extract prompts from YAML
echo "🔄 Extracting prompts from $PROMPTS_FILE..."
PROMPTS_JSON=$(extract_prompts)

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to extract prompts from $PROMPTS_FILE"
    exit 1
fi

# Parse prompts count
PROMPTS_COUNT=$(echo "$PROMPTS_JSON" | python3 -c "import sys, json; data = json.load(sys.stdin); print(len(data))")
echo "✅ Found $PROMPTS_COUNT prompt(s) to fine-tune"
echo ""

# Create base save directory
mkdir -p "$SAVE_DIR_BASE"

# Loop over each prompt
echo "============================================================================"
echo "🔄 Starting fine-tuning for all prompts..."
echo "============================================================================"
echo ""

PROMPT_INDEX=0
echo "$PROMPTS_JSON" | python3 -c "
import sys
import json
import tempfile
import os

data = json.load(sys.stdin)
for i, (key, prompt) in enumerate(data):
    # Write prompt to a temporary file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(prompt)
        temp_file = f.name
    # Output: index, key, and temp file path
    print(f'{i}\t{key}\t{temp_file}')
" | while IFS=$'\t' read -r INDEX KEY PROMPT_FILE; do
    # Skip if any field is empty
    if [ -z "$INDEX" ] || [ -z "$KEY" ] || [ -z "$PROMPT_FILE" ]; then
        continue
    fi

    # Read the prompt from the temporary file
    if [ ! -f "$PROMPT_FILE" ]; then
        echo "⚠️  Warning: Prompt file not found for key: $KEY"
        echo "   Skipping this prompt..."
        continue
    fi

    PROMPT_TEXT=$(cat "$PROMPT_FILE")
    rm -f "$PROMPT_FILE"  # Clean up temp file

    # Check if prompt is empty
    if [ -z "$PROMPT_TEXT" ]; then
        echo "⚠️  Warning: Empty prompt for key: $KEY"
        echo "   Skipping this prompt..."
        continue
    fi

    PROMPT_INDEX=$((INDEX + 1))
    PROMPT_NUM=$(printf "%02d" $PROMPT_INDEX)

    echo ""
    echo "------------------------------------------------------------------------"
    echo "🔄 Fine-tuning prompt $PROMPT_INDEX/$PROMPTS_COUNT: $KEY"
    echo "------------------------------------------------------------------------"
    echo ""

    # Set save directory for this prompt
    SAVE_DIR="${SAVE_DIR_BASE}/prompt_${PROMPT_NUM}_${KEY}"

    # Set HuggingFace repo ID for this prompt
    # Convert key to lowercase and replace underscores/spaces with hyphens
    if [ "$UPLOAD_TO_HF" = true ] && [ -n "$HF_REPO_BASE" ]; then
        KEY_NORMALIZED=$(echo "$KEY" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr ' ' '-')
        HF_REPO_ID="${HF_REPO_BASE}-${KEY_NORMALIZED}"
    else
        HF_REPO_ID=""
    fi

    echo "Prompt: $KEY"
    echo "Save Directory: $SAVE_DIR"
    if [ -n "$HF_REPO_ID" ]; then
        echo "HuggingFace Repo: $HF_REPO_ID"
    fi
    echo ""

    # Write prompt to a temporary file to avoid shell escaping issues
    PROMPT_TEMP_FILE=$(mktemp)
    echo "$PROMPT_TEXT" > "$PROMPT_TEMP_FILE"

    # Build command - use direct execution instead of eval to avoid escaping issues
    if [ "$FP16" = true ]; then
        FP16_FLAG="--fp16"
    else
        FP16_FLAG=""
    fi

    if [ "$UPLOAD_TO_HF" = true ] && [ -n "$HF_REPO_ID" ] && [ -n "$HF_TOKEN" ]; then
        HF_FLAGS="--upload-to-hf --hf-token $HF_TOKEN --repo-id $HF_REPO_ID"
    else
        HF_FLAGS=""
    fi

    # Generate unique WandB run name
    if [ "$USE_WANDB" = true ]; then
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        WANDB_RUN_NAME="${WANDB_RUN_NAME_PREFIX}_${KEY}_${TIMESTAMP}"
        WANDB_FLAGS="--use-wandb --wandb-project $WANDB_PROJECT --wandb-run-name $WANDB_RUN_NAME --wandb-tags finetune $KEY"
    else
        WANDB_FLAGS=""
    fi

    # Run fine-tuning - read prompt from file to avoid quoting issues
    python finetune.py \
        --model-name "$BASE_MODEL" \
        --train-dataset "$TRAIN_DATASET" \
        --eval-dataset "$EVAL_DATASET" \
        --save-dir "$SAVE_DIR" \
        --prompt "$(cat "$PROMPT_TEMP_FILE")" \
        --lora-r $LORA_R \
        --lora-alpha $LORA_ALPHA \
        --lora-dropout $LORA_DROPOUT \
        --learning-rate $LEARNING_RATE \
        --batch-size $BATCH_SIZE \
        --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
        --warmup-ratio $WARMUP_RATIO \
        --max-steps $MAX_STEPS \
        --optim "$OPTIMIZER" \
        --lr-scheduler "$LR_SCHEDULER" \
        --weight-decay $WEIGHT_DECAY \
        --max-seq-length $MAX_SEQ_LENGTH \
        --seed $SEED \
        $FP16_FLAG \
        $HF_FLAGS \
        $WANDB_FLAGS

    # Clean up temp file
    rm -f "$PROMPT_TEMP_FILE"

    echo ""
    echo "✅ Fine-tuning complete for prompt: $KEY"
    echo "   Model saved to: $SAVE_DIR"
    if [ -n "$HF_REPO_ID" ]; then
        echo "   HuggingFace: https://huggingface.co/$HF_REPO_ID"
    fi
    echo ""
done

echo ""
echo "============================================================================"
echo "🎉 All fine-tuning complete!"
echo "============================================================================"
echo ""
echo "Fine-tuned models saved to: $SAVE_DIR_BASE"
echo ""
echo "Next steps:"
echo "  Run './2_inference_and_eval.sh' to evaluate the fine-tuned models"
echo ""
