#!/bin/bash
# ============================================================================
# This script loads the KIE dataset and splits it into train/eval/test sets.
# Run this script FIRST before running any other scripts.
# ============================================================================

set -e  # Stop on any error

DATASET_NAME="nanonets/key_information_extraction"
OUTPUT_DIR="./kie_splits"
TRAIN_RATIO=0.6
EVAL_RATIO=0.2
TEST_RATIO=0.2
SEED=42

# MAIN EXECUTION

echo "============================================================================"
echo "🚀 KIE Data Preparation Script"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  - Dataset: $DATASET_NAME"
echo "  - Output Directory: $OUTPUT_DIR"
echo "  - Train Ratio: $TRAIN_RATIO (60%)"
echo "  - Eval Ratio: $EVAL_RATIO (20%)"
echo "  - Test Ratio: $TEST_RATIO (20%)"
echo "  - Random Seed: $SEED"
echo ""
echo "============================================================================"
echo ""

# Check if splits already exist
if [ -d "$OUTPUT_DIR/train" ] && [ -d "$OUTPUT_DIR/eval" ] && [ -d "$OUTPUT_DIR/test" ]; then
    echo "⚠️  Dataset splits already exist in $OUTPUT_DIR"
    read -p "Do you want to regenerate them? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "✅ Using existing dataset splits."
        exit 0
    fi
    echo "🔄 Regenerating dataset splits..."
fi

# Run data preparation
echo "🔄 Loading and splitting dataset..."
python load_dataset.py \
    --dataset-name "$DATASET_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --train-ratio $TRAIN_RATIO \
    --eval-ratio $EVAL_RATIO \
    --test-ratio $TEST_RATIO \
    --seed $SEED

echo ""
echo "============================================================================"
echo "✅ Data preparation complete!"
echo "============================================================================"
echo ""
echo "Dataset splits saved to: $OUTPUT_DIR"
echo "  - Training set: $OUTPUT_DIR/train"
echo "  - Evaluation set: $OUTPUT_DIR/eval"
echo "  - Test set: $OUTPUT_DIR/test"
echo ""
echo "Next steps:"
echo "  1. Run './1_finetune.sh' to fine-tune models"
echo "  2. Run './2_inference_and_eval.sh' to evaluate models"
echo ""
