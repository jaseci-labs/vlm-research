#!/bin/bash
# ============================================================================
# 1_finetune_random_ex1.sh
#
# Trains ONE model on KIE using MULTIPLE prompts mixed per-sample.
# Prompt assignment: random_quota (exact quotas).
# Writes: prompt_assignment.json
# ============================================================================

set -euo pipefail


# Model configuration
BASE_MODEL="unsloth/gemma-3-12b-it"
TRAIN_DATASET="./kie_splits/train"
EVAL_DATASET="./kie_splits/eval"

# One output model directory
SAVE_DIR="./kie_finetuned/equal_prompt_mix_random_quota"


# Python entrypoint
FINETUNE_PY="finetune_random_ex1.py"
PYTHON_BIN="python3"


# HuggingFace upload
UPLOAD_TO_HF=false
HF_REPO_ID=""   # Add your HuggingFace token here
HF_TOKEN_SCRIPT=""  # Base repo ID (e.g., "Hirudika2002/vlm-finetunes")


# WandB logging
USE_WANDB=false
WANDB_ENTITY="vlm-research"
WANDB_PROJECT="kie-finetuning-random-ex1"
WANDB_RUN_NAME_PREFIX="kie_equalmix_randomquota"


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
PROMPTS_FILE=""
if [ -f "prompts.yaml" ]; then
  PROMPTS_FILE="prompts.yaml"
elif [ -f "prompts.yml" ]; then
  PROMPTS_FILE="prompts.yml"
else
  PROMPTS_FILE="prompts.yaml"
fi

PROMPT_ASSIGNMENT="random_quota"
PROMPT_SEED=3407

PROMPT_ASSIGNMENT_OUT="${SAVE_DIR}/prompt_assignment.json"
export PROMPT_ASSIGNMENT_OUT


# MAIN EXECUTION


echo "============================================================================"
echo "🚀 KIE Fine-tuning Script (ONE RUN - Equal Prompt Mix / Random Quota)"
echo "============================================================================"
echo ""
echo "Model Configuration:"
echo "  - Base Model: $BASE_MODEL"
echo "  - Training Dataset: $TRAIN_DATASET"
echo "  - Evaluation Dataset: $EVAL_DATASET"
echo "  - Save Directory: $SAVE_DIR"
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
echo "Prompts:"
echo "  - Prompts File: $PROMPTS_FILE"
echo "  - Assignment Mode: $PROMPT_ASSIGNMENT"
echo "  - Prompt Seed: $PROMPT_SEED"
echo ""
echo "Upload:"
echo "  - Upload to HF: $UPLOAD_TO_HF"
if [ "$UPLOAD_TO_HF" = true ]; then
  echo "  - Repo: $HF_REPO_ID"
fi
echo ""
echo "WandB Logging:"
echo "  - Enabled: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
  echo "  - Entity: $WANDB_ENTITY"
  echo "  - Project: $WANDB_PROJECT"
fi
echo ""
echo "============================================================================"
echo ""


# Preflight checks

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "❌ Error: $PYTHON_BIN not found."
  exit 1
fi

if [ ! -f "$FINETUNE_PY" ]; then
  echo "❌ Error: $FINETUNE_PY not found."
  exit 1
fi

if [ ! -d "$TRAIN_DATASET" ]; then
  echo "❌ Error: Training dataset not found at $TRAIN_DATASET"
  exit 1
fi

if [ ! -d "$EVAL_DATASET" ]; then
  echo "❌ Error: Evaluation dataset not found at $EVAL_DATASET"
  exit 1
fi

if [ ! -f "$PROMPTS_FILE" ]; then
  echo "❌ Error: Prompts file not found."
  exit 1
fi

mkdir -p "$SAVE_DIR"


# Flags

FP16_FLAG=""
if [ "$FP16" = true ]; then
  FP16_FLAG="--fp16"
fi

HF_TOKEN_VALUE="${HF_TOKEN:-$HF_TOKEN_SCRIPT}"
HF_FLAGS=""
if [ "$UPLOAD_TO_HF" = true ] && [ -n "$HF_REPO_ID" ] && [ -n "$HF_TOKEN_VALUE" ]; then
  HF_FLAGS="--upload-to-hf --hf-token $HF_TOKEN_VALUE --repo-id $HF_REPO_ID"
fi

WANDB_FLAGS=""
if [ "$USE_WANDB" = true ]; then
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  WANDB_RUN_NAME="${WANDB_RUN_NAME_PREFIX}_${TIMESTAMP}"
  WANDB_FLAGS="--use-wandb --wandb-entity $WANDB_ENTITY --wandb-project $WANDB_PROJECT --wandb-run-name $WANDB_RUN_NAME --wandb-tags finetune equalmix random_quota"
fi


# Training 
echo "============================================================================"
echo "🔄 Starting ONE fine-tuning run..."
echo "============================================================================"
echo ""

"$PYTHON_BIN" "$FINETUNE_PY" \
  --model-name "$BASE_MODEL" \
  --train-dataset "$TRAIN_DATASET" \
  --eval-dataset "$EVAL_DATASET" \
  --save-dir "$SAVE_DIR" \
  --prompts-file "$PROMPTS_FILE" \
  --prompt-assignment "$PROMPT_ASSIGNMENT" \
  --prompt-seed "$PROMPT_SEED" \
  --save-prompt-assignment "true" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --learning-rate "$LEARNING_RATE" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --warmup-ratio "$WARMUP_RATIO" \
  --max-steps "$MAX_STEPS" \
  --optim "$OPTIMIZER" \
  --lr-scheduler "$LR_SCHEDULER" \
  --weight-decay "$WEIGHT_DECAY" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --seed "$SEED" \
  $FP16_FLAG \
  $HF_FLAGS \
  $WANDB_FLAGS


# Post-run report

echo ""
echo "============================================================================"
echo "📌 Prompt → Training image indexes (1-based)"
echo "============================================================================"

if [ ! -f "$PROMPT_ASSIGNMENT_OUT" ]; then
  echo "❌ Error: $PROMPT_ASSIGNMENT_OUT not found."
  exit 1
fi

"$PYTHON_BIN" - <<'EOF'
import json, os
path = os.environ.get("PROMPT_ASSIGNMENT_OUT")
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
assignments = data.get("assignments", {})
for k in sorted(assignments):
    idxs = sorted(i + 1 for i in assignments[k])
    print(f"{k} used images indexes={{" + ",".join(map(str, idxs)) + "}}")
EOF

echo ""
echo "============================================================================"
echo "🎉 All done!"
echo "============================================================================"
