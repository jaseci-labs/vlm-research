# Summary & Execution Guide

## Quick Start (TL;DR)

```bash
# Step 1: Prepare data (run once)
./0_prepare_data.sh

# Step 2: Fine-tune models (loops over all prompts in prompts.yml)
./1_finetune.sh

# Step 3: Run inference and evaluation (loops over all prompts)
./2_inference_and_eval.sh

# Step 4: View results
ls -lh kie_results/*.xlsx
```

## Execution Information

### Script 0: Data Preparation

**Run**: `./0_prepare_data.sh`

**What it does**:

- Loads `nanonets/key_information_extraction`
- Splits into train/eval/test (60/20/20)
- Saves to `./kie_splits/`

**Adjust in script** (usually no adjustment needed):

```bash
DATASET_NAME="nanonets/key_information_extraction"
OUTPUT_DIR="./kie_splits"
TRAIN_RATIO=0.6
EVAL_RATIO=0.2
TEST_RATIO=0.2
SEED=42
```

**Output**:

```text
kie_splits/
├── train/    # 60% for fine-tuning
├── eval/     # 20% for validation during training
└── test/     # 20% for final testing
```

---

### Script 1: Fine-tuning

**Run**: `./1_finetune.sh`

**What it does**:

- Loops over all prompts in `prompts.yml`
- Fine-tunes Gemma-3-12B-IT for each prompt
- Saves models to `./kie_finetuned/prompt_XX_<name>/`
- Optionally uploads to HuggingFace

**Adjust in script**:

#### Required Adjustments (for HuggingFace upload)

```bash
UPLOAD_TO_HF=true                          # Set to true to enable
HF_TOKEN="hf_xxxxxxxxxxxxx"                # Your HF token
HF_REPO_BASE="username/kie-gemma3-12b"     # Your HF repo base name
```

#### Optional Adjustments (for fine-tuning)

```bash
MAX_STEPS=100          # Reduce for quick testing (e.g., 10)
BATCH_SIZE=4           # Reduce if OOM (e.g., 2)
LEARNING_RATE=2e-4     # Adjust if needed
```

**Output** (per prompt):

```text
kie_finetuned/
├── prompt_01_baseline/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer files...
│   └── training_stats.json
├── prompt_02_masked/
└── prompt_03_few_shot/
```

---

### Script 2: Inference and Evaluation

**Run**: `./2_inference_and_eval.sh`

**What it does**:

- Loops over all prompts in `prompts.yml`
- Loads corresponding fine-tuned model
- Runs inference on test set
- Calls evaluation script for each prompt
- Saves Excel files with results

**Adjust in script**:

#### For HuggingFace Models

```bash
LOAD_FROM_HF=true                      # Set to true to load from HF
HF_REPO_BASE="username/kie-gemma3-12b" # Your HF repo base name
```

#### For Sample Selection

```bash
SAMPLE_INDICES="0,1,2,3,4,5,6,7,8,9"  # Which test samples to evaluate
# For quick testing: "0,1,2"
# For full test set: "0,1,2,3,...,N"
```

#### For WandB Logging

```bash
USE_WANDB=true             # Set to true to enable
WANDB_PROJECT="kie-eval"   # Your WandB project name
```

---

## Adding New Prompts

Add new prompt:

```yaml
your_prompt_name:
  name: "Your Prompt Description"
  text: |
    Your detailed prompt text here.
    Can be multi-line.
    Will be used for both training and inference.
```

The scripts automatically detect and process all prompts in `prompts.yml`!
