#!/usr/bin/env python3
"""
Evaluate Flickr30k predictions using CIDEr, SPICE, and Cosine similarity metrics.

- Computes average score over 5 references by default.
- Optionally, compute maximum score per sample using `--use-max-ref`.
- Logs results to Excel and optionally to WandB.

Install:
    pip install git+https://github.com/salaniz/pycocoevalcap
    pip install pandas pillow xlsxwriter tqdm datasets sentence-transformers
"""

from __future__ import annotations
import argparse
import json
from io import BytesIO
from typing import List, Dict, Tuple, Any

import copy
from collections import defaultdict
import math

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from tqdm import tqdm

# SPICE from pycocoevalcap
from pycocoevalcap.spice.spice import Spice

# sentence-transformers for Cosine similarity 
from sentence_transformers import SentenceTransformer, util


# CIDEr implementation 

def precook(s: str, n: int = 4):
    """
    Convert a sentence string into a dict of n-gram counts.
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n: int = 4):
    """
    Apply precook to a list of reference sentences.
    """
    return [precook(ref, n) for ref in refs]

def cook_test(test, n: int = 4):
    """
    Apply precook to a single test (prediction) sentence.
    """
    return precook(test, n)

class CiderScorer(object):
    """
    CIDEr scorer as in your cider.py:
    - builds TF-IDF vectors from n-gram counts
    - computes cosine similarity between prediction and references
    """

    def copy(self):
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        new.document_frequency = copy.copy(self.document_frequency)
        new.ref_len = self.ref_len
        return new

    def __init__(self, test=None, refs=None, n: int = 4, sigma: float = 6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []                 # list of list of ref n-gram dicts
        self.ctest = []                 # list of test n-gram dicts
        self.document_frequency = defaultdict(float)
        self.ref_len = None
        self.cook_append(test, refs)

    def cook_append(self, test, refs):
        """
        Add one (test, refs) pair, where:
          - test: string
          - refs: list of strings
        """
        if refs is not None:
            self.crefs.append(cook_refs(refs, n=self.n))
            if test is not None:
                self.ctest.append(cook_test(test, n=self.n))
            else:
                self.ctest.append(None)

    def size(self):
        assert len(self.crefs) == len(self.ctest), f"refs/test mismatch! {len(self.crefs)}<>{len(self.ctest)}"
        return len(self.crefs)

    def __iadd__(self, other):
        """
        Support += with either:
          - (test_string, list_of_ref_strings)
          - another CiderScorer
        """
        if isinstance(other, tuple):
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self):
        """
        Compute document frequency of each n-gram across all references.
        """
        for refs in self.crefs:
            # refs is a list of dicts (n-gram -> count)
            ngram_set = set()
            for ref in refs:
                for ngram in ref.keys():
                    ngram_set.add(ngram)
            for ngram in ngram_set:
                self.document_frequency[ngram] += 1.0

    def compute_cider(self):
        """
        Compute CIDEr scores (one per (test, refs) pair added).
        """
        def counts2vec(cnts):
            # cnts: dict of n-gram -> count
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]

            for (ngram, term_freq) in cnts.items():
                df = math.log(max(1.0, self.document_frequency.get(ngram, 0.0)))
                n = len(ngram) - 1
                if n >= self.n:
                    continue
                # TF-IDF weight
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                norm[n] += vec[n][ngram] ** 2
                if n == 1:
                    length += term_freq

            norm = [math.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            val = np.zeros(self.n)
            for n in range(self.n):
                for (ngram, _) in vec_hyp[n].items():
                    val[n] += vec_hyp[n][ngram] * vec_ref[n].get(ngram, 0.0)
                if norm_hyp[n] != 0 and norm_ref[n] != 0:
                    val[n] /= (norm_hyp[n] * norm_ref[n])
                assert not math.isnan(val[n])
            return val

        # This matches your cider.py: fixed ref_len based on COCO.
        self.ref_len = math.log(float(40504))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.zeros(self.n)
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)

        return scores


# Utilities

def safe_to_text(x: Any) -> str:
    """
    Convert any reference object to a simple text string suitable for metrics.
    Handles strings, dicts with common keys, lists, etc. Falls back to str().
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        # join list elements into a string
        try:
            return " ".join(str(i) for i in x)
        except Exception:
            return str(x)
    if isinstance(x, dict):
        # common patterns
        if "caption" in x and isinstance(x["caption"], str):
            return x["caption"]
        if "caption" in x:
            return str(x["caption"])
        if "sentence" in x and isinstance(x["sentence"], str):
            return x["sentence"]
        if "raw" in x and isinstance(x["raw"], str):
            return x["raw"]
        # fallback: stringify dict
        return " ".join(str(v) for v in x.values())
    # fallback
    return str(x)

def preprocess_caption(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.strip().lower()


# CIDEr computation (now using inlined CiderScorer)

def compute_cider_scores(
    predictions: List[str],
    references: List[List[Any]],
    average_over_refs: bool = True
) -> Tuple[float, List[float]]:
    """
    Compute corpus CIDEr and per-sample CIDEr scores using the inlined CiderScorer.

    - average_over_refs=True  -> standard CIDEr over all references.
    - average_over_refs=False -> for each sample, compute CIDEr vs each single
      reference and take the max over refs. Corpus score is mean of these max scores.
    """

    # Average-over-references CIDEr (standard) 
    cider_scorer = CiderScorer(n=4)

    for pred, ref_list in zip(predictions, references):
        pred_pp = preprocess_caption(safe_to_text(pred))
        ref_pp = [preprocess_caption(safe_to_text(r)) for r in ref_list]
        # add (prediction_string, list_of_reference_strings)
        cider_scorer += (pred_pp, ref_pp)

    # Build document frequency from all references
    cider_scorer.compute_doc_freq()
    # Per-sample CIDEr scores
    per_sample_scores = cider_scorer.compute_cider()
    per_sample_scores = [float(s) for s in per_sample_scores]
    corpus_score = float(np.mean(per_sample_scores)) if per_sample_scores else 0.0

    if average_over_refs:
        return corpus_score, per_sample_scores

    # Max-over-references CIDEr 
    per_sample_max = []
    for pred, ref_list in zip(predictions, references):
        pred_pp = preprocess_caption(safe_to_text(pred))
        sample_scores = []

        for r in ref_list:
            single_ref_pp = [preprocess_caption(safe_to_text(r))]

            scorer = CiderScorer(n=4)
            scorer += (pred_pp, single_ref_pp)
            scorer.compute_doc_freq()
            s_list = scorer.compute_cider()  # list with one element

            if s_list:
                sample_scores.append(float(s_list[0]))
            else:
                sample_scores.append(0.0)

        per_sample_max.append(max(sample_scores) if sample_scores else 0.0)

    corpus_max = float(np.mean(per_sample_max)) if per_sample_max else 0.0
    return corpus_max, per_sample_max


# SPICE computation 

def compute_spice_scores(
    predictions: List[str],
    references: List[List[Any]],
    average_over_refs: bool = True
) -> Tuple[float, List[float]]:
    """
    Compute SPICE corpus and per-sample scores.

    SPICE expects:
      gts: {id: [ref_str1, ref_str2, ...]}
      res: {id: [candidate_str]}

    - average_over_refs=True:
        Use all references per sample (standard SPICE).
    - average_over_refs=False:
        For each sample, compute SPICE vs each single ref separately,
        and take the max F-score over refs.
    """
    # Standard SPICE: all refs per sample 
    gts = {}
    res = {}
    for i, (pred, ref_list) in enumerate(zip(predictions, references)):
        pred_pp = preprocess_caption(safe_to_text(pred))
        ref_pps = [preprocess_caption(safe_to_text(r)) for r in ref_list]
        gts[i] = ref_pps                # list of strings
        res[i] = [pred_pp]              # list of one string

    spice_scorer = Spice()
    try:
        corpus_score, per_instance = spice_scorer.compute_score(gts, res)
        per_sample_scores: List[float] = []
        for inst in per_instance:
            # inst is usually a dict with "All" -> {"f": ...}
            if isinstance(inst, dict) and "All" in inst and "f" in inst["All"]:
                val = inst["All"]["f"]
                per_sample_scores.append(float(val) if val is not None else 0.0)
            else:
                # fallback (in case implementation differs)
                try:
                    per_sample_scores.append(float(inst))
                except Exception:
                    per_sample_scores.append(0.0)
    except Exception as e:
        print(f"❌ Error computing SPICE (multi-ref): {e}")
        corpus_score = 0.0
        per_sample_scores = [0.0 for _ in predictions]

    if average_over_refs:
        return float(corpus_score), per_sample_scores

    # Max-over-references SPICE
    per_sample_max: List[float] = []
    for pred, ref_list in zip(predictions, references):
        pred_pp = preprocess_caption(safe_to_text(pred))
        sample_scores: List[float] = []

        for r in ref_list:
            ref_pp = preprocess_caption(safe_to_text(r))
            gts_single = {0: [ref_pp]}
            res_single = {0: [pred_pp]}

            try:
                c, inst_list = spice_scorer.compute_score(gts_single, res_single)
                if isinstance(inst_list, (list, tuple)) and len(inst_list) > 0:
                    inst = inst_list[0]
                    if isinstance(inst, dict) and "All" in inst and "f" in inst["All"]:
                        val = inst["All"]["f"]
                        sample_scores.append(float(val) if val is not None else 0.0)
                    else:
                        sample_scores.append(float(c))
                else:
                    sample_scores.append(float(c))
            except Exception as e:
                print(f"❌ Error computing SPICE (single-ref): {e}")
                sample_scores.append(0.0)

        per_sample_max.append(max(sample_scores) if sample_scores else 0.0)

    corpus_max = float(sum(per_sample_max) / len(per_sample_max)) if per_sample_max else 0.0
    return corpus_max, per_sample_max

# Cosine similarity computation 
def compute_cosine_scores(predictions: List[str], references: List[List[Any]], average_over_refs: bool = True) -> Tuple[float, List[float]]:
    """
    Compute sentence-embedding cosine similarity between each prediction and its references.
    Uses sentence-transformers 'all-MiniLM-L6-v2'.
    Returns corpus_score (average of per-sample scores) and list of per-sample scores.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    per_sample_scores = []

    for pred, ref_list in zip(predictions, references):
        pred_text = preprocess_caption(safe_to_text(pred))
        refs_texts = [preprocess_caption(safe_to_text(r)) for r in ref_list]
        if not refs_texts:
            per_sample_scores.append(0.0)
            continue
        pred_emb = model.encode(pred_text, convert_to_tensor=True)
        ref_embs = model.encode(refs_texts, convert_to_tensor=True)
        sims = util.cos_sim(pred_emb, ref_embs).cpu().numpy().flatten()
        if average_over_refs:
            per_sample_scores.append(float(sims.mean()))
        else:
            per_sample_scores.append(float(sims.max()))

    corpus_score = float(sum(per_sample_scores) / len(per_sample_scores)) if per_sample_scores else 0.0
    return corpus_score, per_sample_scores

#  Excel Logging 
def log_metrics_to_excel(
    sample_indices: List[int],
    predictions: Dict[int, str],
    ground_truths: Dict[int, List[str]],
    cider_scores: List[float],
    spice_scores: List[float],
    cosine_scores: List[float],
    inference_times: Dict[int, float],
    vram_usage: Dict[int, float],
    test_dataset,
    prompt: str,
    model_name: str,
    output_excel_path: str = "flickr30k_eval_results.xlsx",
):
    """
    Log evaluation metrics to an Excel file with images.
    """

    rows = []
    pil_images = []

    for i, idx in enumerate(sample_indices):
        pred = predictions.get(idx, "")
        gt_list = ground_truths.get(idx, [])
        time_taken = inference_times.get(idx, 0.0)
        vram = vram_usage.get(idx, 0.0)
        cider = cider_scores[i] if i < len(cider_scores) else 0.0
        spice = spice_scores[i] if i < len(spice_scores) else 0.0
        cosine = cosine_scores[i] if i < len(cosine_scores) else 0.0

        sample_item = test_dataset[idx]
        pil_img = sample_item.get("image") if isinstance(sample_item, dict) else sample_item["image"]

        if isinstance(pil_img, bytes):
            pil_img = PILImage.open(BytesIO(pil_img))
        elif not isinstance(pil_img, PILImage.Image):
            try:
                import numpy as _np
                pil_img = PILImage.fromarray(_np.asarray(pil_img))
            except Exception:
                pil_img = None

        pil_images.append(pil_img)
        gt_text = "\n".join(str(c) for c in gt_list)

        row = {
            "sample_index": idx,
            "prompt": prompt,
            "model": model_name,
            "ground_truths": gt_text,
            "prediction": pred,
            "cider_score": cider,
            "spice_score": spice,
            "cosine_score": cosine,
            "inference_time_s": time_taken,
            "vram_usage_mb": vram,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    writer = pd.ExcelWriter(output_excel_path, engine="xlsxwriter")
    sheet_name = "evaluation"
    df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=1, index=False)
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    for row_idx, pil_img in enumerate(pil_images, start=1):
        if pil_img is None:
            continue
        try:
            stream = BytesIO()
            pil_img.thumbnail((128, 128))
            pil_img.save(stream, format="PNG")
            stream.seek(0)
            worksheet.insert_image(row_idx, 0, f"img_{row_idx}.png", {"image_data": stream})
        except Exception:
            pass

    # adjust columns a bit
    worksheet.set_column(0, 0, 20)
    worksheet.set_column(1, 1, 12)
    worksheet.set_column(2, 2, 50)
    worksheet.set_column(3, 3, 20)
    worksheet.set_column(4, 4, 60)
    worksheet.set_column(5, 5, 15)
    worksheet.set_column(6, 6, 15)
    worksheet.set_column(7, 7, 15)
    worksheet.set_column(8, 8, 12)
    worksheet.set_column(9, 9, 15)
    worksheet.set_column(10, 10, 15)

    writer.close()
    print(f"✅ Excel written to: {output_excel_path}")

# WandB Logging 
def log_metrics_to_wandb(
    sample_indices,
    predictions,
    ground_truths,
    cider_scores,
    spice_scores,
    cosine_scores,
    inference_times,
    vram_usage,
    test_dataset,
    prompt,
    model_name,
    project_name,
    run_name=None,
):
    try:
        import wandb
    except ImportError:
        print("⚠️ WandB not installed — skipping.")
        return

    run = wandb.init(project=project_name, name=run_name, reinit=True)

    table = wandb.Table(columns=[
        "sample_index", "image", "prediction", "ground_truths",
        "cider_score", "spice_score", "cosine_score",
        "inference_time_s", "vram_usage_mb", "prompt"
    ])

    for i, idx in enumerate(sample_indices):
        pred = predictions.get(idx, "")
        gt_list = ground_truths.get(idx, [])
        gt_str = "\n".join(gt_list)
        cider = cider_scores[i] if i < len(cider_scores) else 0.0
        spice = spice_scores[i] if i < len(spice_scores) else 0.0
        cosine = cosine_scores[i] if i < len(cosine_scores) else 0.0

        sample_item = test_dataset[idx]
        pil_img = sample_item.get("image") if isinstance(sample_item, dict) else sample_item["image"]

        if isinstance(pil_img, bytes):
            pil_img = PILImage.open(BytesIO(pil_img))
        elif not isinstance(pil_img, PILImage.Image):
            try:
                import numpy as _np
                pil_img = PILImage.fromarray(_np.asarray(pil_img))
            except Exception:
                pil_img = None

        wb_img = wandb.Image(pil_img) if pil_img else None

        table.add_data(
            idx, wb_img, pred, gt_str, cider, spice, cosine,
            inference_times.get(idx, 0.0),
            vram_usage.get(idx, 0.0),
            prompt,
        )

    wandb.log({"evaluation_table": table})
    wandb.finish()
    print("✅ Logged to WandB")

# MAIN 
if __name__ == "__main__":
    import datasets

    parser = argparse.ArgumentParser(description="Evaluate Flickr30k predictions using CIDEr, SPICE, and Cosine similarity")

    # Input
    parser.add_argument("--inference-results", type=str, required=True,
                        help="Path to inference results JSON file")
    parser.add_argument("--test-dataset", type=str, required=True,
                        help="Path to Flickr30k test dataset")

    # Output
    parser.add_argument("--output-excel", type=str, default="flickr30k_eval_results.xlsx",
                        help="Output Excel file path")
    parser.add_argument("--model-name", type=str, default="my-model",
                        help="Model name for logging")

    # WandB
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="flickr30k-eval",
                        help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="WandB run name (optional)")

    # Max vs average references
    parser.add_argument("--use-max-ref", action="store_true", default=False,
                        help="If set, compute max per-sample score over 5 references (default: average)")

    args = parser.parse_args()

    # Load results
    with open(args.inference_results, "r") as f:
        results = json.load(f)
    predictions = {int(k): v for k, v in results.get("predictions", {}).items()}
    ground_truths = {int(k): v for k, v in results.get("ground_truths", {}).items()}
    inference_times = {int(k): v for k, v in results.get("inference_times", {}).items()}
    vram_usage = {int(k): v for k, v in results.get("vram_usage", {}).items()}
    sample_indices = results.get("sample_indices", [])
    prompt = results.get("prompt", "")

    # Prepare lists
    pred_list = [predictions[idx] for idx in sample_indices]
    gt_list = [ground_truths[idx] for idx in sample_indices]
    average_over_refs = not args.use_max_ref

    # Compute metrics
    print("\n🔄 Running evaluation (CIDEr + SPICE + Cosine)...")
    corpus_cider, per_sample_cider = compute_cider_scores(pred_list, gt_list, average_over_refs)
    corpus_spice, per_sample_spice = compute_spice_scores(pred_list, gt_list, average_over_refs)
    corpus_cosine, per_sample_cosine = compute_cosine_scores(pred_list, gt_list, average_over_refs)

    print(f"✅ Evaluation complete!")
    print(f"   - CIDEr (corpus): {corpus_cider:.4f}")
    print(f"   - SPICE (corpus): {corpus_spice:.4f}")
    print(f"   - Cosine (corpus): {corpus_cosine:.4f}")

    # Load dataset for images
    test_dataset = datasets.load_from_disk(args.test_dataset)

    # Log to Excel
    log_metrics_to_excel(
        sample_indices, predictions, ground_truths,
        per_sample_cider, per_sample_spice, per_sample_cosine,
        inference_times, vram_usage, test_dataset, prompt,
        args.model_name, args.output_excel
    )

    # Log to WandB
    if args.use_wandb:
        log_metrics_to_wandb(
            sample_indices, predictions, ground_truths,
            per_sample_cider, per_sample_spice, per_sample_cosine,
            inference_times, vram_usage, test_dataset, prompt,
            args.model_name, args.wandb_project, args.wandb_run_name
        )

    print("\n✅ Evaluation complete!")
