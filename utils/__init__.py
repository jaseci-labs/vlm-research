#!/usr/bin/env python3
"""
VLM Research - Shared Utilities Package

This package provides common utilities used across VLM research experiments:
- loaders: Model and dataset loading functions
- metrics: Evaluation metrics (KIE, Levenshtein, etc.)
- formatting: Text/conversation formatting utilities
- inference: Inference pipeline utilities
- registry: Experiment discovery and dispatch
- wandb_utils: Weights & Biases integration
"""

from .loaders import (
    load_model,
    load_base_model_for_training,
    load_and_split_dataset,
    load_dataset_from_disk,
    decode_image,
)

from .metrics import (
    Field,
    GroundTruth,
    Prediction,
    get_kie_metrics,
    evaluate_kie_predictions,
    compute_levenshtein_similarity,
)

from .formatting import (
    clean_json_output,
    convert_to_conversation,
    convert_kie_sample_to_conversation,
    build_chat_messages,
    format_dict_as_json,
    parse_json_safely,
)

from .inference import (
    run_inference,
    run_inference_batch,
    get_gpu_stats,
)

from .registry import (
    load_experiment_registry,
    get_experiment_config,
    list_experiments,
    get_experiment_path,
    get_experiment_scripts,
    print_registry_summary,
)

from .wandb_utils import (
    DEFAULT_WANDB_PROJECT,
    init_wandb,
    log_metrics,
    log_metrics_to_excel,
    log_metrics_to_wandb,
    finish_wandb,
)

__all__ = [
    # Loaders
    "load_model",
    "load_base_model_for_training",
    "load_and_split_dataset",
    "load_dataset_from_disk",
    "decode_image",
    # Metrics
    "Field",
    "GroundTruth",
    "Prediction",
    "get_kie_metrics",
    "evaluate_kie_predictions",
    "compute_levenshtein_similarity",
    # Formatting
    "clean_json_output",
    "convert_to_conversation",
    "convert_kie_sample_to_conversation",
    "build_chat_messages",
    "format_dict_as_json",
    "parse_json_safely",
    # Inference
    "run_inference",
    "run_inference_batch",
    "get_gpu_stats",
    # Registry
    "load_experiment_registry",
    "get_experiment_config",
    "list_experiments",
    "get_experiment_path",
    "get_experiment_scripts",
    "print_registry_summary",
    # WandB
    "DEFAULT_WANDB_PROJECT",
    "init_wandb",
    "log_metrics",
    "log_metrics_to_excel",
    "log_metrics_to_wandb",
    "finish_wandb",
]
