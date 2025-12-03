# Investigation: CUDA Initialization Failure on 5th Inference Iteration

**Date:** December 3, 2025  
**Environment:** RunPod container, NVIDIA A40 (46GB VRAM), CUDA 12.9  
**Script:** `2_inference_and_eval.sh`  
**Status:** 🔍 Under Investigation

## Issue Summary

During execution of `2_inference_and_eval.sh`, the script successfully completed inference and evaluation for **4 out of 5 prompts**. On the **5th iteration** (prompt "noisy"), the process failed with a CUDA initialization error despite the GPU being physically available.

## Error Output

```bash
------------------------------------------------------------------------
🔄 Evaluating prompt 5/5: noisy
------------------------------------------------------------------------

Prompt: noisy
Model Path: unsloth/gemma-3-12b-it
Inference Output: ./kie_results/prompt_05_noisy
Evaluation Excel: ./kie_results/KIE_prompt_05_noisy.xlsx

🔄 Running inference...
/workspace/vlm-research/experiments/prompting/kie/unsloth_env/lib/python3.11/site-packages/torch/cuda/__init__.py:182:
UserWarning: CUDA initialization: CUDA driver initialization failed, you might not have a CUDA gpu.
(Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:119.)
  return torch._C._cuda_getDeviceCount() > 0

Traceback (most recent call last):
  File "/workspace/vlm-research/experiments/prompting/kie/inference.py", line 15, in <module>
    from unsloth import FastVisionModel
  File ".../unsloth/__init__.py", line 80, in <module>
    import unsloth_zoo
  File ".../unsloth_zoo/__init__.py", line 136, in <module>
    from .device_type import (
  File ".../unsloth_zoo/device_type.py", line 56, in <module>
    DEVICE_TYPE : str = get_device_type()
  File ".../unsloth_zoo/device_type.py", line 46, in get_device_type
    raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
```

## Post-Failure GPU Status

Immediately after the failure, `nvidia-smi` confirmed the GPU was healthy and idle:

```bash
(unsloth_env) root@6af983d65562:/workspace/vlm-research/experiments/prompting/kie# nvidia-smi
Wed Dec  3 14:07:01 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.64.03              Driver Version: 575.64.03      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A40                     On  |   00000000:57:00.0 Off |                    0 |
|  0%   31C    P8             33W /  300W |       0MiB /  46068MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

**Key Observation:** GPU hardware is fine (0% util, 0MiB used, 31°C). The issue is software/driver-level, not hardware.

## Root Cause Analysis

### Primary Hypothesis: CUDA Context Corruption

The error `CUDA driver initialization failed` indicates the CUDA context (within the Python process or driver communication layer) became corrupted after multiple heavy model loads.

### Contributing Factors

| Factor                      | Description                                                                                | Likelihood  |
| --------------------------- | ------------------------------------------------------------------------------------------ | ----------- |
| **CUDA Context Corruption** | Repeated model loading/unloading without full cleanup corrupts CUDA state                  | ⭐⭐⭐ HIGH |
| **Memory Fragmentation**    | Memory not released cleanly between iterations causing "zombie" allocations                | ⭐⭐ MEDIUM |
| **Driver Instability**      | Long sessions with multiple CUDA context initializations can trigger driver bugs           | ⭐⭐ MEDIUM |
| **Process Isolation Issue** | Shell script spawns separate Python processes, but parent shell may retain corrupted state | ⭐ LOW      |

### Why Only the 5th Iteration?

- Each inference run loads a 12B parameter model (~13GB VRAM)
- 4 consecutive load/unload cycles accumulated residual state
- By iteration 5, the corruption threshold was reached
- The GPU was "available" but the CUDA driver context was in an invalid state

## Testing & Verification

### Test 1: Verify GPU Availability

```bash
nvidia-smi
```

**Result:** ✅ GPU available, healthy, idle

### Test 2: Run Failed Prompt in Isolation

```bash
python inference.py \
    --base-model "unsloth/gemma-3-12b-it" \
    --test-dataset "./kie_splits/test" \
    --prompt "$(python3 -c "import yaml; print(yaml.safe_load(open('prompts.yml'))['noisy']['text'])")" \
    --max-new-tokens 256 \
    --output-dir "./kie_results/prompt_05_noisy"
```

**Expected Result:** Should work in fresh process

### Test 3: Run Evaluation After Successful Inference

```bash
python evaluate.py \
    --inference-results "./kie_results/prompt_05_noisy/inference_results.json" \
    --test-dataset "./kie_splits/test" \
    --output-excel "./kie_results/KIE_prompt_05_noisy.xlsx" \
    --model-name "gemma-3-12b-it"
```

### Test 4: Full Pipeline Retry with Fix

After applying the resolution below, re-run:

```bash
./2_inference_and_eval.sh
```

## Resolution

### Fix 1: Add GPU Cleanup in `inference.py`

Add explicit garbage collection and CUDA cache clearing at the end of `run_inference_batch()`:

```python
# At the end of run_inference_batch(), before the return statement:

import gc

# Force cleanup to prevent CUDA issues in long-running loops
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

return predictions, ground_truths, inference_times, vram_usage
```

### Fix 2: Add Inter-Iteration Cleanup in Shell Script

Add a Python cleanup step between iterations in `2_inference_and_eval.sh`:

```bash
# After each inference completes, add:
python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; import gc; gc.collect()"
```

### Fix 3: Environment Reset (Nuclear Option)

If issues persist, reset CUDA environment:

```bash
# Reset CUDA context before each iteration
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -r  # Reset GPU (requires root)
```
