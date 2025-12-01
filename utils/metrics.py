#!/usr/bin/env python3
"""
Shared evaluation metrics for VLM research experiments.
"""

import json
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

try:
    from Levenshtein import distance as edit_distance
except ImportError:
    edit_distance = None


# ----- KIE Metric Data Structures -----

class Field:
    """Represents a single KIE field with label and value."""

    def __init__(self, label: str, value: str):
        self.label = label
        self.value = value

    def __repr__(self):
        return f"Field(label='{self.label}', value='{self.value}')"


class GroundTruth:
    """Container for ground truth fields."""

    def __init__(self, fields: List[Field]):
        self.fields = fields

    def __repr__(self):
        return f"GroundTruth(fields={self.fields})"


class Prediction:
    """Container for prediction with ground truth reference."""

    def __init__(self, fields: List[Field], gt: GroundTruth):
        self.fields = fields
        self.gt = gt

    def _get_pred_field_by_label(self, label: str) -> Optional[Field]:
        """Get prediction field by label name."""
        for field in self.fields:
            if field.label == label:
                return field
        return None

    def __repr__(self):
        return f"Prediction(fields={self.fields}, gt={self.gt})"


# ----- KIE Metrics -----

def get_kie_metrics(predictions: List[Prediction]) -> Tuple[float, List[float]]:
    """
    Compute Levenshtein similarity per prediction (sample-level),
    averaged across fields.

    Args:
        predictions: List of Prediction objects with ground truth

    Returns:
        Tuple of (overall_average_score, per_sample_scores)

    Example:
        >>> avg_score, scores = get_kie_metrics(predictions)
        >>> print(f"Average KIE accuracy: {avg_score:.4f}")
    """
    if edit_distance is None:
        raise ImportError(
            "python-Levenshtein is required for KIE metrics. "
            "Install with: pip install python-Levenshtein"
        )

    sample_scores = []

    for pred in tqdm(predictions, desc="Computing KIE metrics", leave=False):
        gt_fields = pred.gt.fields
        field_scores = []

        for gt_field in gt_fields:
            pred_field = pred._get_pred_field_by_label(gt_field.label)
            if pred_field is None or pred_field.value == "":
                pred_value = ""
            else:
                pred_value = pred_field.value

            pred_value = str(pred_value)
            gt_value = str(gt_field.value)

            dist = edit_distance(pred_value, gt_value)
            max_len = max(len(pred_value), len(gt_value))
            if max_len == 0:
                field_scores.append(1.0)
            else:
                field_scores.append(1 - (dist / max_len))

        sample_avg = sum(field_scores) / len(field_scores) if field_scores else 0.0
        sample_scores.append(sample_avg)

    overall_avg = sum(sample_scores) / len(sample_scores) if sample_scores else 0.0
    return overall_avg, sample_scores


def evaluate_kie_predictions(
    preds: List[str],
    gts: List[Dict]
) -> Tuple[float, List[float]]:
    """
    High-level evaluation wrapper for KIE predictions.
    Handles JSON parsing and creates Prediction objects.

    Args:
        preds: List of prediction strings (JSON format expected)
        gts: List of ground truth dicts

    Returns:
        Tuple of (average_accuracy, per_sample_scores)

    Example:
        >>> predictions = ['{"date": "2024-01-15", "total": "100.00"}', ...]
        >>> ground_truths = [{"date": "2024-01-15", "total": "100.00"}, ...]
        >>> avg_score, scores = evaluate_kie_predictions(predictions, ground_truths)
    """
    assert len(preds) == len(gts), \
        "Predictions and ground truth lists must be the same length."

    prediction_objects = []

    for pred_raw, gt_json in zip(preds, gts):
        # Parse prediction if it's a string
        if isinstance(pred_raw, str):
            try:
                pred_json = json.loads(pred_raw)
            except json.JSONDecodeError:
                # If parsing fails, treat as empty prediction
                pred_json = {}
        else:
            pred_json = pred_raw

        # Parse ground truth if it's a string
        if isinstance(gt_json, str):
            try:
                gt_json = json.loads(gt_json)
            except json.JSONDecodeError:
                gt_json = {}

        gt_fields = [Field(label, value) for label, value in gt_json.items()]
        pred_fields = [Field(label, pred_json.get(label, "")) for label in gt_json.keys()]

        prediction = Prediction(fields=pred_fields, gt=GroundTruth(gt_fields))
        prediction_objects.append(prediction)

    return get_kie_metrics(prediction_objects)


def compute_levenshtein_similarity(pred: str, gt: str) -> float:
    """
    Compute Levenshtein similarity between two strings.
    Returns a score between 0 and 1, where 1 is exact match.

    Args:
        pred: Prediction string
        gt: Ground truth string

    Returns:
        Similarity score (0.0 to 1.0)
    """
    if edit_distance is None:
        raise ImportError(
            "python-Levenshtein is required. "
            "Install with: pip install python-Levenshtein"
        )

    pred = str(pred)
    gt = str(gt)

    dist = edit_distance(pred, gt)
    max_len = max(len(pred), len(gt))

    if max_len == 0:
        return 1.0

    return 1 - (dist / max_len)
