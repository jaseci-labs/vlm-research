# Hand Conded Prompt Evaluation

This folder contains evaluation scripts and Jupyter notebooks for testing and comparing visual language model (VLM) captioning performance across two datasets: **CAR Damage Dataset (CARDD)** and **Flickr30k**. It includes automated metric evaluation using CIDEr, SPICE, and cosine similarity.

## 📂 Files Overview

### `cardd_pixtral.ipynb`

Notebook to evaluate image captioning outputs from the **Pixtral VLM** on the CAR Damage Dataset (CARDD).

- Loads CARDD images and prompts Pixtral for caption generation.
- Logs outputs, inference time, and GPU usage.
- Evaluates caption quality using CIDEr, SPICE, and cosine similarity.
- Outputs results to Excel and logs visual comparisons using Weights & Biases (W&B).

### `cardd_qwen.ipynb`

Similar to `cardd_pixtral.ipynb`, but uses the **Qwen-VL** model for inference on the CARDD dataset.

- Follows the same workflow for prompt-based captioning and metric evaluation.

### `flickr30k_pixtral.ipynb`

Runs the **Pixtral VLM** on the Flickr30k dataset using similar structured prompting and evaluation.

- Useful for comparing model performance on a more general-purpose image-caption dataset.
- Designed to replicate the CARDD evaluation flow but with different image and caption sources.

### `flickr30k_qwen.ipynb`

Captioning and evaluation of Flickr30k images using the **Qwen-VL** model.

- Follows the same pipeline as `flickr30k_pixtral.ipynb` but replaces the model for side-by-side comparisons.

## ⚙️ How to Run the Notebooks

1. **Install dependencies**:
   Make sure you have the required packages installed:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up project directory**:

   - Place image datasets under `/workspace/data/`
   - For CARDD: `test_dataset/` and `test_set.json`
   - For Flickr30k: `filtered_dataset/`
   - Add `cardd-df.p` and `flickr-df.p` to same folder for cider evaluation.

3. **Execute notebook**:

   - Open either notebook (e.g., `cardd_qwen.ipynb`) in Jupyter or VSCode.
   - Run cells sequentially.
   - Output Excel file and W\&B logs will be generated automatically.

4. **Outputs**:

   - Caption predictions and metrics are saved in the format:

     - `prompt_tuning_results_cardd_qwen.xlsx`, etc.
   - Visual logs pushed to your [Weights & Biases](https://wandb.ai/vlm-research) project.

## 📧 Contact

Built by [Gayanuka Amarasuriya](https://gayanukaa.github.io/).
