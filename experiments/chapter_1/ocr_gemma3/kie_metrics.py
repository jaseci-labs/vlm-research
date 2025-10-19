from __future__ import annotations
from typing import List, Union
from Levenshtein import distance as edit_distance
from tqdm import tqdm
import json

# ----- Define helper data structures -----

class Field:
    def __init__(self, label: str, value: str):
        self.label = label
        self.value = value

class GroundTruth:
    def __init__(self, fields: List[Field]):
        self.fields = fields

class Prediction:
    def __init__(self, fields: List[Field], gt: GroundTruth):
        self.fields = fields
        self.gt = gt

    def _get_pred_field_by_label(self, label: str):
        for field in self.fields:
            if field.label == label:
                return field
        return None

# ----- Scoring function -----

def get_kie_metrics(predictions: List[Prediction]) -> tuple[float, List[float]]:
    """
    Compute Levenshtein similarity per prediction (sample-level),
    averaged across fields, and return:
        - overall average score
        - list of per-sample scores
    """
    sample_scores = []

    for pred in tqdm(predictions, desc="Computing KIE metrics", leave=False):
        gt_fields = pred.gt.fields
        field_scores = []

        for gt_field in gt_fields:
            pred_field = pred._get_pred_field_by_label(gt_field.label)
            if pred_field is None or pred_field == "":
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

# ----- Evaluation wrapper -----

def evaluate_kie_predictions(preds: List[Union[str, dict]], gts: List[dict]) -> tuple[float, List[float]]:
    """
    Evaluate KIE predictions against ground truth using Levenshtein similarity.

    Returns:
        - Average accuracy score across all samples.
        - List of individual sample-level accuracy scores.
    """
    assert len(preds) == len(gts), "Predictions and ground truth lists must be the same length."

    prediction_objects = []

    for pred_raw, gt_json in zip(preds, gts):
        if isinstance(pred_raw, str):
            pred_json = json.loads(pred_raw, strict=False)
        else:
            pred_json = pred_raw

        gt_fields = [Field(label, value) for label, value in gt_json.items()]
        pred_fields = [Field(label, pred_json.get(label, "")) for label in gt_json.keys()]

        prediction = Prediction(fields=pred_fields, gt=GroundTruth(gt_fields))
        prediction_objects.append(prediction)

    return get_kie_metrics(prediction_objects)
