import wandb
import numpy as np
import pandas as pd

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from sentence_transformers import SentenceTransformer, util


# ----------------------------------------------------------------------------
# wandb configuration and fetching artifacts
# ----------------------------------------------------------------------------
wandb.login()
run = wandb.init(project="Model Performance Comparison", name="Standard Metrics-Exp-2", entity="vlm-research")

artifact_qwen = wandb.use_artifact("vlm-research/Model Performance Comparison/run-7r2ypykd-ImageJSONTable-420fe573a41f724e6b1a3d1bccfed10f:v0")
table_qwen = artifact_qwen.get("Image JSON Table")

artifact_gemma = wandb.use_artifact("vlm-research/Model Performance Comparison/run-khp3a3d3-predictions_table:v0")
table_gemma = artifact_gemma.get("predictions_table")

# ----------------------------------------------------------------------------
# Load ground truth captions from CSV
# ----------------------------------------------------------------------------
csv_path = "random_3000_rows.csv"
df_gt = pd.read_csv(csv_path)

gt_map = {}
for _, row in df_gt.iterrows():
    wandb_img_path = "image_dataset/" + row['filename']
    if wandb_img_path not in gt_map:
        gt_map[wandb_img_path] = []
    gt_map[wandb_img_path].append(str(row['captions']))

# ----------------------------------------------------------------------------
# Functions to format data for evaluation metrics
# ----------------------------------------------------------------------------
def format_for_pycocoevalcap(candidates, references_lists):
    gts = {}
    res = {}
    if len(candidates) != len(references_lists):
        raise ValueError("Number of candidates and reference sets must be equal.")
    for i in range(len(candidates)):
        sample_id = str(i)
        gts[sample_id] = []
        for ref_text in references_lists[i]:
            gts[sample_id].append({"caption": ref_text})
        res[sample_id] = [{"caption": candidates[i]}]
    return gts, res

def extract_model_response(response):
    """
    Extracts the first string enclosed in double quotes or curly quotes from the model's response.
    """
    import re
    match = re.search(r'["“](.+?)["”]', response)
    if match:
        return match.group(1).strip()
    return response.strip()


# ----------------------------------------------------------------------------
# Functions to calculate CIDER, Cosine Similarity, and SPICE
# ----------------------------------------------------------------------------
def calculate_cider(candidates, references_lists):
    gts, res = format_for_pycocoevalcap(candidates, references_lists)
    scorer = Cider()
    gts_cider = {k: [item['caption'] for item in v] for k, v in gts.items()}
    res_cider = {k: [item['caption'] for item in v] for k, v in res.items()}
    score, scores_per_instance = scorer.compute_score(gts_cider, res_cider)
    return score, scores_per_instance


def calculate_cosine_similarity_st(candidates, references_lists, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    similarities = []
    for cand, refs in zip(candidates, references_lists):
        if not refs:
            similarities.append(0)
            continue
        cand_embedding = model.encode(cand, convert_to_tensor=True)
        ref_embeddings = model.encode(refs, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(cand_embedding, ref_embeddings)[0]
        cos_scores_numpy = cos_scores.cpu().numpy()
        similarities.append(np.mean(cos_scores_numpy))
    overall_avg_similarity = np.mean(similarities) if similarities else 0
    return overall_avg_similarity, similarities


def calculate_spice(candidates, references_lists, stanford_corenlp_home=None):
    gts, res = format_for_pycocoevalcap(candidates, references_lists)
    gts_spice = {k: [item['caption'] for item in v] for k, v in gts.items()}
    res_spice = {k: [item['caption'] for item in v] for k, v in res.items()}
    if stanford_corenlp_home:
        print(f"Temporarily set STANFORD_CORENLP_HOME to: {stanford_corenlp_home}")
    scorer = Spice()
    try:
        score, scores_per_instance = scorer.compute_score(gts_spice, res_spice)
    except Exception as e:
        print(f"Error calculating SPICE: {e}")
        return None, None
    return score, scores_per_instance


def max_metric_over_refs(candidate, references, cider_scorer, spice_scorer, st_model, stanford_corenlp_home=None):
    if not references:
        return 0.0, 0.0, 0.0
    
    # CIDER
    cider_scores = []
    for ref in references:
        _, scores = cider_scorer.compute_score({0: [ref]}, {0: [candidate]})
        cider_scores.append(scores[0])
    max_cider = max(cider_scores)

    # SPICE
    spice_scores = []
    for ref in references:
        try:
            _, scores = spice_scorer.compute_score({0: [ref]}, {0: [candidate]})
            spice_scores.append(scores[0]['All']['f'])
        except Exception:
            spice_scores.append(0.0)
    max_spice = max(spice_scores)

    # Cosine Similarity
    cand_emb = st_model.encode(candidate, convert_to_tensor=True)
    ref_embs = st_model.encode(references, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(cand_emb, ref_embs)[0].cpu().numpy()
    max_cosine = float(np.max(cos_scores))
    return float(max_cider), float(max_spice), float(max_cosine)


# ----------------------------------------------------------------------------
# Main execution block
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    qwen_data = table_qwen.data[:200]
    gemma_data = table_gemma.data[:200]

    qwen_responses = []
    gemma_responses = []
    image_paths = []
    images = []

    for row_qwen, row_gemma in zip(qwen_data, gemma_data):
        qwen_raw = row_qwen[table_qwen.columns.index("teacher_response")]
        gemma_raw = row_gemma[table_gemma.columns.index("prediction")]

        gemma_response = extract_model_response(gemma_raw)
        qwen_response = extract_model_response(qwen_raw)

        image_path = row_qwen[table_qwen.columns.index("image_path")]
        image = row_qwen[table_qwen.columns.index("image")]

        qwen_responses.append(qwen_response)
        gemma_responses.append(gemma_response)
        image_paths.append(image_path)
        images.append(image)

    # Prepare ground truth captions for each image
    ground_truth_lists = [gt_map.get(path, []) for path in image_paths]

    # Prepare qwen as reference for gemma-vs-qwen
    qwen_refs = [[ref] for ref in qwen_responses]

    # Prepare scorers/models
    cider_scorer = Cider()
    spice_scorer = Spice()
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    stanford_path = "/workspace/nlp_tools/stanford-corenlp-4.5.10"

    # Calculate scores
    qwen_gt_cider, qwen_gt_spice, qwen_gt_cosine = [], [], []
    gemma_gt_cider, gemma_gt_spice, gemma_gt_cosine = [], [], []
    gemma_qwen_cider, gemma_qwen_spice, gemma_qwen_cosine = [], [], []

    for i in range(len(image_paths)):
        gt_captions = ground_truth_lists[i]
        qwen_resp = qwen_responses[i]
        gemma_resp = gemma_responses[i]
        # Qwen vs GT
        c, s, cos = max_metric_over_refs(qwen_resp, gt_captions, cider_scorer, spice_scorer, st_model, stanford_corenlp_home=stanford_path)
        qwen_gt_cider.append(c)
        qwen_gt_spice.append(s)
        qwen_gt_cosine.append(cos)
        # Gemma vs GT
        c, s, cos = max_metric_over_refs(gemma_resp, gt_captions, cider_scorer, spice_scorer, st_model, stanford_corenlp_home=stanford_path)
        gemma_gt_cider.append(c)
        gemma_gt_spice.append(s)
        gemma_gt_cosine.append(cos)
    # Gemma vs Qwen (single reference)
    # Use batch scoring for efficiency
    gemma_qwen_cider_score, gemma_qwen_cider_scores = calculate_cider(gemma_responses, qwen_refs)
    _, gemma_qwen_cosine_scores = calculate_cosine_similarity_st(gemma_responses, qwen_refs)
    _, gemma_qwen_spice_scores = calculate_spice(gemma_responses, qwen_refs, stanford_corenlp_home=stanford_path)
    gemma_qwen_spice_f = [instance['All']['f'] for instance in gemma_qwen_spice_scores] if gemma_qwen_spice_scores else [0.0]*len(gemma_responses)

    # ----------------------------------------------------------------------------
    # Build and log the new table
    # ----------------------------------------------------------------------------
    new_columns = [
        "image_path", "image", "ground_truth_captions", "qwen_response", "gemma_response",
        "qwen_gt_cider", "qwen_gt_spice", "qwen_gt_cosine",
        "gemma_gt_cider", "gemma_gt_spice", "gemma_gt_cosine",
        "gemma_qwen_cider", "gemma_qwen_spice", "gemma_qwen_cosine"
    ]
    new_table = wandb.Table(columns=new_columns)
    for i in range(len(image_paths)):
        new_table.add_data(
            image_paths[i],
            images[i],
            ground_truth_lists[i],
            qwen_responses[i],
            gemma_responses[i],
            float(qwen_gt_cider[i]),
            float(qwen_gt_spice[i]),
            float(qwen_gt_cosine[i]),
            float(gemma_gt_cider[i]),
            float(gemma_gt_spice[i]),
            float(gemma_gt_cosine[i]),
            float(gemma_qwen_cider_scores[i]),
            float(gemma_qwen_spice_f[i]),
            float(gemma_qwen_cosine_scores[i])
        )
    run.log({"evaluation_results_v2": new_table})
    run.finish()