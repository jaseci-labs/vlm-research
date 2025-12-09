#!/bin/bash
# ============================================================================
# This script runs inference on fine-tuned models and evaluates them
# on the Flickr30k dataset using CIDEr, SPICE, and Cosine similarity.
# Loops over all prompts in prompts.yml and evaluates each corresponding model.
#
# Usage:
#   - For LOCAL models: Leave HF_MODEL_REPO empty, models loaded from BASE_MODEL
#   - For HF adapter models: Set HF_MODEL_REPO to the full repository path
# ============================================================================
set -e  # Stop on any error

# ------------------------- CONFIGURATION -------------------------
BASE_MODEL="unsloth/gemma-3-12b-it"
TEST_DATASET="./flickr30k_splits/test"
MODEL_DIR_BASE="./flickr30k_finetuned"
OUTPUT_DIR="./flickr30k_results"

# HuggingFace adapter repo (leave empty if not using)
HF_MODEL_REPO=""

# Inference parameters
MAX_NEW_TOKENS=256
SAMPLE_INDICES=""  # Empty = all samples, or specify comma-separated indices like "0,1,2,3"

# Evaluation parameters
MODEL_NAME="gemma-3-12b-it"
USE_WANDB=false
WANDB_ENTITY="vlm-research"
WANDB_PROJECT="flickr-inferencing"
WANDB_RUN_NAME_PREFIX="base_model_inference"

# Prompts file
PROMPTS_FILE="prompts.yml"

# Optional: use max over references instead of average
USE_MAX_REF=false  # Default false (average)

# ------------------------- HELPER FUNCTIONS -------------------------
extract_prompts() {
    python3 << 'EOF'
import yaml, json, sys

try:
    with open("prompts.yml", "r") as f:
        data = yaml.safe_load(f)
    prompts = [(k, v["text"].strip()) for k, v in data.items() if isinstance(v, dict) and "text" in v]
    print(json.dumps(prompts))
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

# ------------------------- MAIN EXECUTION -------------------------
echo "============================================================================"
echo "🚀 Flickr30k Inference and Evaluation (CIDEr + SPICE + Cosine)"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  - Base Model: $BASE_MODEL"
echo "  - Flickr30k Test Dataset: $TEST_DATASET"
echo "  - Output Directory: $OUTPUT_DIR"
if [ -n "$HF_MODEL_REPO" ]; then
    echo "  - HuggingFace Adapter Repo: $HF_MODEL_REPO"
fi
echo ""
echo "Inference Parameters:"
echo "  - Max New Tokens: $MAX_NEW_TOKENS"
echo "  - Decoding: Greedy (deterministic)"
if [ -n "$SAMPLE_INDICES" ]; then
    echo "  - Sample Indices: $SAMPLE_INDICES"
else
    echo "  - Sample Indices: ALL (entire test dataset)"
fi
echo ""
echo "Evaluation Parameters:"
echo "  - Model Name: $MODEL_NAME"
echo "  - Use WandB: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
    echo "  - WandB Project: $WANDB_PROJECT"
    echo "  - WandB Run Name Prefix: $WANDB_RUN_NAME_PREFIX"
fi
echo "  - Use Max Ref: $USE_MAX_REF (default=false = average over refs)"
echo ""
echo "============================================================================"
echo ""

# Check if test dataset exists
if [ ! -d "$TEST_DATASET" ]; then
    echo "❌ Error: Flickr30k test dataset not found at $TEST_DATASET"
    echo "Please prepare the dataset (e.g., with datasets.load_dataset(...).save_to_disk)."
    exit 1
fi

# Check if prompts file exists
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "❌ Error: Prompts file not found at $PROMPTS_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Extract prompts from YAML
echo "🔄 Extracting prompts from $PROMPTS_FILE..."
PROMPTS_JSON=$(extract_prompts)

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to extract prompts from $PROMPTS_FILE"
    exit 1
fi

# Parse prompts count
PROMPTS_COUNT=$(echo "$PROMPTS_JSON" | python3 -c "import sys, json; data = json.load(sys.stdin); print(len(data))")
echo "✅ Found $PROMPTS_COUNT prompt(s) to evaluate"
echo ""

# Loop over each prompt
echo "============================================================================"
echo "🔄 Starting inference and evaluation for all prompts..."
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
    echo "🔄 Evaluating prompt $PROMPT_INDEX/$PROMPTS_COUNT: $KEY"
    echo "------------------------------------------------------------------------"
    echo ""

    # Determine model path
    if [ -n "$HF_MODEL_REPO" ]; then
        MODEL_PATH="$HF_MODEL_REPO"
        LOAD_FROM_HF_FLAG="--load-from-hf"
    else
        MODEL_PATH="$BASE_MODEL"
        LOAD_FROM_HF_FLAG=""
    fi

    INFERENCE_OUTPUT_DIR="${OUTPUT_DIR}/prompt_${PROMPT_NUM}_${KEY}"
    INFERENCE_RESULTS="${INFERENCE_OUTPUT_DIR}/inference_results.json"
    EVAL_EXCEL="${OUTPUT_DIR}/flickr30k_prompt_${PROMPT_NUM}_${KEY}.xlsx"

    mkdir -p "$INFERENCE_OUTPUT_DIR"

    echo "Prompt key: $KEY"
    echo "Model Path: $MODEL_PATH"
    echo "Inference Output: $INFERENCE_OUTPUT_DIR"
    echo "Evaluation Excel: $EVAL_EXCEL"
    echo ""

    # ========================================================================
    # Step 1: Run Inference
    # ========================================================================

    echo "🔄 Running inference on Flickr30k..."

    # Write prompt to a temporary file to avoid shell escaping issues
    PROMPT_TEMP_FILE=$(mktemp)
    echo "$PROMPT_TEXT" > "$PROMPT_TEMP_FILE"

    python Inference.py \
        --model-path "$MODEL_PATH" \
        $LOAD_FROM_HF_FLAG \
        --base-model "$BASE_MODEL" \
        --test-dataset "$TEST_DATASET" \
        --prompt "$(cat "$PROMPT_TEMP_FILE")" \
        --sample-indices "$SAMPLE_INDICES" \
        --max-new-tokens $MAX_NEW_TOKENS \
        --output-dir "$INFERENCE_OUTPUT_DIR"

    rm -f "$PROMPT_TEMP_FILE"

    echo "✅ Inference complete!"
    echo ""

    # ========================================================================
    # Step 2: Run Evaluation (CIDEr + SPICE + Cosine)
    # ========================================================================

    echo "🔄 Running evaluation (CIDEr + SPICE + Cosine)..."

    # Generate unique WandB run name
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    WANDB_RUN_NAME="${WANDB_RUN_NAME_PREFIX}_${KEY}_${TIMESTAMP}"

    EVAL_CMD="python evaluate.py \
        --inference-results \"$INFERENCE_RESULTS\" \
        --test-dataset \"$TEST_DATASET\" \
        --output-excel \"$EVAL_EXCEL\" \
        --model-name \"$MODEL_NAME\""

    # Add WandB flags if enabled
    if [ "$USE_WANDB" = true ]; then
        EVAL_CMD="$EVAL_CMD --use-wandb --wandb-project \"$WANDB_PROJECT\" --wandb-run-name \"$WANDB_RUN_NAME\""
    fi

    # Add max-ref flag if USE_MAX_REF is true
    if [ "$USE_MAX_REF" = true ]; then
        EVAL_CMD="$EVAL_CMD --use-max-ref"
    fi

    eval $EVAL_CMD

    echo "✅ Evaluation complete! Results saved to: $EVAL_EXCEL"
    echo ""
done

echo ""
echo "============================================================================"
echo "🎉 All Flickr30k inference and evaluation complete!"
echo "============================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary of outputs:"
ls -lh "$OUTPUT_DIR"/*.xlsx 2>/dev/null || echo "No Excel files found."
echo ""
echo "You can now review the CIDEr, SPICE, and Cosine scores in the Excel files."
echo ""
