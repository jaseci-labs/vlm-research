# Visual Language Model Prompt Evaluation

This folder presents a comparative evaluation of prompting methods methods applied to image captioning tasks using visual language models (VLMs).

The evaluation spans across datasets such as **CAR Damage Dataset (CARDD)** and **Flickr30k**, and utilizes key metrics like CIDEr, SPICE, and cosine similarity.

## 👨‍💻 Contributors & Approaches

| Method        | Contributor | GitHub Profile |
|---------------|-------------|----------------|
| HC Only       | Gayanuka Amarasuriya | [@Gayanukaa](https://github.com/Gayanukaa) |
| FT Only       | Ushari Vidanage       | [@SarangaVP](https://github.com/SarangaVP)   |
| HC + FT       | Saranga Abeywickrama  | [@ushariRanasinghe](https://github.com/ushariRanasinghe) |

## 📁 Folder Structure & Usage

Each method is implemented in its own subdirectory. Please refer to the respective folders for full implementation details, notebooks, and execution instructions:

- **HC Only**: [HC Evaluation Folder](./hc-only/)  
- **FT Only**: [FT Evaluation Folder](./ft-only/)  
- **HC + FT**: [HC + FT Evaluation Folder](./hc+ft/)

Each folder includes:

- VLM-based captioning notebooks (`*_pixtral.ipynb`, `*_qwen.ipynb`)
- Visual comparison logging via Weights & Biases
- Output logs to `.xlsx` files
- Custom prompt evaluation logic

## 📊 `cider.py`

This script implements the **CIDEr** metric used for evaluating how similar generated captions are to ground-truth captions.

- Adapted for local use to compute CIDEr scores without relying on external APIs.
- Also used alongside SPICE and cosine similarity as part of the `evaluate_all_metrics()` function inside the notebooks.

## 📐 Metrics Used

- **CIDEr**: Measures consensus between generated and reference captions based on TF-IDF weighting of n-grams.
- **SPICE**: Semantic propositional content metric based on scene graph tuples.
- **Cosine Similarity**: Embedding-based semantic comparison between reference and generated captions.

## 📎 Notes

- Ensure images matched to the filenames in your JSON caption files.
- All models assume prompts are evaluated with **6 templates**, each run **twice**, for a total of **12 outputs per image**.
- Results can be used for **comparative analysis of prompting strategies** and **model performance across domains**.
