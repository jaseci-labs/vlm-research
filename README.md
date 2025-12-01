# VLM Research Experiment Framework

<!--
## 📄 Paper
**Title:** [Your Paper Title Here]
**Authors:** [Author Names]
**Conference/Journal:** [Venue, Year]

```bibtex
@article{your_citation_key,
  title={Your Paper Title},
  author={Author Names},
  journal={Journal/Conference},
  year={2025}
}
```
-->

A modular, future-proof experiment framework for research and development on Visual Language Models (VLMs). Ideal for conducting prompting experiments, fine-tuning comparisons, training image impact studies, and more.

## Quick Start

### Installation

```bash
git clone https://github.com/Jaseci-Labs/vlm-research.git
cd vlm-research
pip install -r requirements.txt
```

## Configuration

### Models

| Config       | Base Model                     | HuggingFace Link                                            |
| ------------ | ------------------------------ | ----------------------------------------------------------- |
| `gemma3_12b` | `unsloth/gemma-3-12b-it`       | [Link](https://huggingface.co/unsloth/gemma-3-12b-it)       |
| `gemma3_4b`  | `unsloth/gemma-3-4b-it`        | [Link](https://huggingface.co/unsloth/gemma-3-4b-it)        |
| `qwen7b`     | `unsloth/Qwen2-VL-7B-Instruct` | [Link](https://huggingface.co/unsloth/Qwen2-VL-7B-Instruct) |

### Datasets

| Config       | Dataset                    | HuggingFace Link                                                                                           |
| ------------ | -------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `kie`        | Key Information Extraction | [nanonets/key_information_extraction](https://huggingface.co/datasets/nanonets/key_information_extraction) |
| `flickr`     | Flickr30k Image Captioning | [nlphuji/flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k)                                     |
| `car_damage` | Car Damage Assessment      | Custom                                                                                                     |
| `rsicd`      | Remote Sensing Captioning  | Custom                                                                                                     |

## Development

### Adding a New Experiment

1. Create folder: `experiments/<your_experiment>/`
2. Add scripts: `finetune.py`, `inference.py`, `evaluate.py`
3. Register in `experiment_registry.yaml`
4. (Optional) Add config overrides in `configs/`

### Shared Utilities

Import from `utils/` package:

```python
from utils import (
    load_model,
    load_and_split_dataset,
    evaluate_kie_predictions,
    run_inference,
    init_wandb,
)
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) - Efficient VLM training
- [Hugging Face](https://huggingface.co/) - Models and datasets
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
