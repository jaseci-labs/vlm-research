# KIE Pipeline - Master Summary & Execution Guide

## 🎯 Quick Start (TL;DR)

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

---

## 📁 New Files Created

### Shell Scripts

1. ✅ **`0_prepare_data.sh`** - Loads and splits dataset (60/20/20)
2. ✅ **`1_finetune.sh`** - Fine-tunes model for each prompt in prompts.yml
3. ✅ **`2_inference_and_eval.sh`** - Runs inference + evaluation for each prompt

### Python Files

1. ✅ **`load_dataset.py`** - Dataset loading and splitting logic
2. ✅ **`finetune.py`** - Fine-tuning implementation
3. ✅ **`inference.py`** - Inference implementation
4. ✅ **`evaluate.py`** - Evaluation metrics and logging

### Documentation

1. ✅ **`README_NEW.md`** - Comprehensive documentation (read this for details)
2. ✅ **`QUICK_REFERENCE.md`** - Quick reference guide (read this for quick help)
3. ✅ **`CHANGES_SUMMARY.md`** - Complete change log (read this to see what changed)
4. ✅ **`MASTER_SUMMARY.md`** - This file (execution guide)

---

## 🔑 Key Changes Made

### 1. Model Switch

- **Before**: `unsloth/Qwen2-VL-7B-Instruct`
- **After**: `unsloth/gemma-3-12b-it`

### 2. Dataset Splits

- **Before**: 80% train, 20% test
- **After**: 60% train, 20% eval, 20% test (no overlap)

### 3. Hyperparameters Updated

- **LoRA**: r=8, alpha=8, dropout=0.01
- **Training**: LR=2e-4, batch=4, optimizer=Adam, warmup=10%, schedule=cosine, fp16
- **Inference**: max_tokens=256, greedy decoding

### 4. Structure Reorganization

- **Before**: 1 monolithic script doing everything
- **After**: 3 independent scripts with clear separation of concerns

---

## 📋 Execution Order & What to Adjust

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

```
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

#### Required Adjustments (for HuggingFace upload):

```bash
UPLOAD_TO_HF=true                          # Set to true to enable
HF_TOKEN="hf_xxxxxxxxxxxxx"                # Your HF token
HF_REPO_BASE="username/kie-gemma3-12b"     # Your HF repo base name
```

#### Optional Adjustments (for fine-tuning):

```bash
MAX_STEPS=100          # Reduce for quick testing (e.g., 10)
BATCH_SIZE=4           # Reduce if OOM (e.g., 2)
LEARNING_RATE=2e-4     # Adjust if needed
```

**Output** (per prompt):

```
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

#### For HuggingFace Models:

```bash
LOAD_FROM_HF=true                      # Set to true to load from HF
HF_REPO_BASE="username/kie-gemma3-12b" # Your HF repo base name
```

#### For Sample Selection:

```bash
SAMPLE_INDICES="0,1,2,3,4,5,6,7,8,9"  # Which test samples to evaluate
# For quick testing: "0,1,2"
# For full test set: "0,1,2,3,...,N"
```

#### For WandB Logging:

```bash
USE_WANDB=true             # Set to true to enable
WANDB_PROJECT="kie-eval"   # Your WandB project name
```

**Output** (per prompt):

```
kie_results/
├── prompt_01_baseline/
│   └── inference_results.json
├── KIE_prompt_01_baseline.xlsx    # Excel with images and metrics
├── prompt_02_masked/
├── KIE_prompt_02_masked.xlsx
├── prompt_03_few_shot/
└── KIE_prompt_03_few_shot.xlsx
```

---

## 🎨 Adding New Prompts

### Step 1: Edit `prompts.yml`

Add your new prompt:

```yaml
your_prompt_name:
  name: "Your Prompt Description"
  text: |
    Your detailed prompt text here.
    Can be multi-line.
    Will be used for both training and inference.
```

### Step 2: Run Scripts

```bash
# Fine-tune for new prompt (will process all prompts)
./1_finetune.sh

# Evaluate new prompt (will process all prompts)
./2_inference_and_eval.sh
```

The scripts automatically detect and process all prompts in `prompts.yml`!

---

## 🔧 Common Adjustments

### For Quick Testing

```bash
# In 1_finetune.sh:
MAX_STEPS=10
BATCH_SIZE=2

# In 2_inference_and_eval.sh:
SAMPLE_INDICES="0,1,2"
```

### For Memory Issues (OOM)

```bash
# In 1_finetune.sh:
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=2

# In 2_inference_and_eval.sh:
MAX_NEW_TOKENS=128
```

### For Full Deployment

```bash
# In 1_finetune.sh:
UPLOAD_TO_HF=true
HF_TOKEN="your_token"
HF_REPO_BASE="username/repo"

# In 2_inference_and_eval.sh:
LOAD_FROM_HF=true
HF_REPO_BASE="username/repo"
USE_WANDB=true
WANDB_PROJECT="your-project"
```

---

## 📊 Understanding the Results

### Excel Output Format

Each Excel file contains:

| Column | Content          | Description                     |
| ------ | ---------------- | ------------------------------- |
| A      | Image            | Thumbnail (128x128) of document |
| B      | sample_index     | Index in test set               |
| C      | prompt           | Prompt used for this model      |
| D      | model            | Model name                      |
| E      | ground_truth     | Ground truth JSON               |
| F      | prediction       | Model's predicted JSON          |
| G      | kie_score        | KIE score (0.0-1.0)             |
| H      | inference_time_s | Time in seconds                 |
| I      | vram_usage_mb    | VRAM in MB                      |

### KIE Score

- **Metric**: Levenshtein edit distance
- **Range**: 0.0 (worst) to 1.0 (perfect)
- **Calculation**: Average across all fields in the JSON
- **Higher is better**: 1.0 means exact match

---

## 🚨 Troubleshooting

### Error: "Dataset not found"

**Solution**: Run `./0_prepare_data.sh` first

### Error: "Model not found"

**Solution**:

- If using local: Run `./1_finetune.sh` first
- If using HF: Set `LOAD_FROM_HF=true` in script 2

### Out of Memory (OOM)

**Solution**: Reduce `BATCH_SIZE` or `MAX_NEW_TOKENS`

### Script won't run

**Solution**: Make it executable: `chmod +x *.sh`

### WandB error

**Solution**: Login first: `wandb login`

### No output files

**Solution**: Check script output for errors

---

## 📚 Which Documentation to Read?

1. **Just want to run it?** → This file (MASTER_SUMMARY.md)
2. **Need quick help?** → QUICK_REFERENCE.md
3. **Want full details?** → README_NEW.md
4. **What changed?** → CHANGES_SUMMARY.md

---

## ✅ Verification Checklist

Before running, verify:

- [ ] Dataset exists or will be downloaded
- [ ] `prompts.yml` has at least one prompt
- [ ] Scripts are executable (`chmod +x *.sh`)
- [ ] Enough disk space for models (~10GB per prompt)
- [ ] Enough GPU memory (12GB+ recommended)
- [ ] HF token set if uploading to HuggingFace
- [ ] WandB login if using WandB

---

## 🎓 Best Practices

1. **Start small**: Test with 1-2 prompts and `MAX_STEPS=10`
2. **Monitor GPU**: Use `watch -n 1 nvidia-smi` during training
3. **Save models**: Keep fine-tuned models before re-running
4. **Version prompts**: Track `prompts.yml` changes in git
5. **Document experiments**: Use WandB or keep notes
6. **Backup results**: Excel files are overwritten on re-run

---

## 🔄 Typical Workflow

### First Time Setup

```bash
# 1. Prepare environment
chmod +x unslothinstall.sh
./unslothinstall.sh
source unsloth_env/bin/activate
pip install PyYAML python-Levenshtein xlsxwriter openpyxl

# 2. Prepare data
./0_prepare_data.sh
```

### Training Iteration

```bash
# 1. Edit prompts.yml (add/modify prompts)
nano prompts.yml

# 2. Fine-tune
./1_finetune.sh

# 3. Evaluate
./2_inference_and_eval.sh

# 4. Review results
ls -lh kie_results/*.xlsx
```

### Experiment Tracking

```bash
# For each experiment:
# 1. Document changes to prompts.yml
# 2. Run fine-tuning
# 3. Run evaluation with WandB enabled
# 4. Save/rename result files
# 5. Compare metrics
```

---

## 📞 Quick Reference Links

- **Dataset**: https://huggingface.co/datasets/nanonets/key_information_extraction
- **Base Model**: https://huggingface.co/unsloth/gemma-3-12b-it
- **Unsloth**: https://github.com/unslothai/unsloth

---

## 🎉 You're All Set!

The pipeline is ready to use. Follow the execution order above and adjust variables as needed. Each script is independent and well-documented. Happy fine-tuning! 🚀

---

**Created**: October 19, 2025  
**Status**: ✅ Complete and tested  
**Scripts**: 3 independent shell scripts  
**Python Files**: 4 modular implementations  
**Documentation**: 4 comprehensive guides
