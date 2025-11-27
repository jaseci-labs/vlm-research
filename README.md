# VLM Research Experiment Framework

A modular, future-proof experiment framework for research and development on **Visual Language Models (VLMs)**. Built with **Hydra**, **Weights & Biases**, and extensible YAML-based configurations — ideal for conducting prompting experiments, fine-tuning comparisons, training image impact studies, and more.

---
<!--
## Features

- Modular config-based experiment control (via [Hydra](https://hydra.cc))
- W&B integration for grouped runs, logging, and comparisons
- Supports **PEFT vs full fine-tuning**, image ablation, and prompting studies
- Clean architecture for easily extending models, datasets, and tasks
- Hydra sweep support for systematic hyperparameter tuning
- Ready for scaling to large experiments
-->
---

## Project Structure

```text
vlm-research/
├── main.py                            # Lightweight experiment dispatcher
├── experiment_registry.yaml           # Maps experiment name -> experiments/<name>/run.py
│
├── experiments/                       # Each experiment is isolated and self-contained
│   ├── peft_vs_full/
│   │   ├── run.py                     # Entry point for this experiment
│   │   ├── logic.py                   # Core training/eval logic
│   │   ├── config.yaml                # Local override config (Hydra)
│   │   ├── outputs/
│   │   └── logs/                      # Hydra logs and model artifacts
│   │
│   └──  prompt_eval/
│       ├── run.py
│       ├── evaluator.py
│       ├── config.yaml
│       ├── outputs/
│       └── logs/                      # Hydra logs and model artifacts
│
├── configs/                           # Global Hydra configs (model, dataset, sweep)
│   ├── config.yaml
│   ├── model/
│   ├── training/
│   ├── dataset/
│   └── sweep/
│
├── core/                              # Shared utilities
│   ├── wandb_utils.py
│   ├── loaders.py
│   ├── metrics.py
│   └── registry.py                    # Load experiment modules dynamically
│
├── README.md
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/yourusername/vlm-research.git
cd vlm-research
pip install -r requirements.txt
```

---

## Running Experiments

### Run a single experiment

```bash
python main.py experiment=peft_vs_full model=qwen7b dataset=car_damage
```

### Run a hyperparameter sweep (Hydra multirun)

```bash
python main.py -m sweep.lr=1e-5,5e-5,1e-4 sweep.batch_size=4,8,16
```

Or with YAML:

```bash
python main.py -m +sweep=lr_vs_batch
```

---

## W&B Logging

Each run will automatically log:

- Metrics
- Config values
- Grouped by experiment type
- Custom run names (e.g., `qwen7b_1e-5_8batch`)

W&B project and experiment name are customizable via config.

---

## Adding New Experiments

1. Add new YAML to `configs/experiment/`
2. Update `experiment_registry.yaml`
3. Add handling logic to `runner.py`

---
<!--
## Ideal Use Cases

- VLM-based insurance prediction
- Prompt vs. finetuning comparison studies
- Training data ablation experiments
- Academic/research reproducibility

---

## TODO / Coming Soon

- [ ] Hugging Face model auto-loading
- [ ] Fine-tuning via Unsloth or LoRA
- [ ] Built-in evaluation metrics for VQA, OCR, etc.
- [ ] Frontend integration with Jac Vision UI
- [ ] On-the-fly model training via backend API

---

## Contributing

We're building this as a flexible backend for the **Jac Vision** no-code platform and for visual AI experimentation. If you're a researcher, engineer, or builder — PRs and suggestions are welcome.

---

## 📄 License

MIT — free to use, extend, and share.
-->
