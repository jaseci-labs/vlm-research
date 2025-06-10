import wandb
import numpy as np

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from sentence_transformers import SentenceTransformer, util


wandb.login()
run = wandb.init(project="Model Performance Comparison", name="Standard Metrics-Exp-2", entity="vlm-research")

artifact_qwen = wandb.use_artifact("vlm-research/Model Performance Comparison/run-7r2ypykd-ImageJSONTable-420fe573a41f724e6b1a3d1bccfed10f:v0")
table_qwen = artifact_qwen.get("Image JSON Table")

artifact_gemma = wandb.use_artifact("vlm-research/Model Performance Comparison/run-khp3a3d3-predictions_table:v0")
table_gemma = artifact_gemma.get("predictions_table")

new_columns = table_qwen.columns + ["gemma_response", "cider_score", "spice_score", "cosine_similarity"]
new_table = wandb.Table(columns=new_columns)


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
        print("Ensure Stanford CoreNLP is correctly set up (jars accessible, sufficient memory).")
        print("Try setting SPICE_JAR and STANFORD_CORENLP_MODELS_JAR environment variables.")
        print("Or, place stanford-corenlp-X.X.X.jar and stanford-corenlp-X.X.X-models.jar")
        print("in pycocoevalcap/spice/lib/ (you might need to create this path).")
        return None, None
    return score, scores_per_instance


def extract_model_response(response):
    """
    Extracts the first string enclosed in double quotes or curly quotes from the model's response.
    """
    import re
    # Match text inside standard or curly double quotes (handles **"..."**, **“...”**, etc.)
    match = re.search(r'["“](.+?)["”]', response)
    if match:
        return match.group(1).strip()
    return response.strip()


if __name__ == "__main__":
    # Only use the first 200 rows from Qwen and Gemma tables
    qwen_data = table_qwen.data[:200]
    gemma_data = table_gemma.data[:200]

    qwen_responses = []
    gemma_responses = []

    for row_qwen, row_gemma in zip(qwen_data, gemma_data):
        qwen_response = row_qwen[table_qwen.columns.index("teacher_response")]
        gemma_raw = row_gemma[table_gemma.columns.index("prediction")]
        gemma_response = extract_model_response(gemma_raw)
        qwen_responses.append(qwen_response)
        gemma_responses.append(gemma_response)
        
    candidates = gemma_responses
    references_lists = [[ref] for ref in qwen_responses]

    print("--- Calculating CIDER ---")
    cider_score, cider_scores_per_instance = calculate_cider(candidates, references_lists)
    if cider_score is not None:
        print(f"Overall CIDER score: {cider_score:.4f}")

    print("\n--- Calculating Cosine Similarity (Sentence Transformers) ---")
    avg_cos_sim_st, cos_sim_st_per_instance = calculate_cosine_similarity_st(candidates, references_lists)
    cos_sim_st_per_instance_python = [float(x) for x in cos_sim_st_per_instance]
    print(f"Overall Average Cosine Similarity (Sentence Transformers): {avg_cos_sim_st:.4f}")
    
    print("\n--- Calculating SPICE ---")
    stanford_path = "/workspace/nlp_tools/stanford-corenlp-4.5.9"
    spice_score, spice_scores_per_instance = calculate_spice(candidates, references_lists, stanford_corenlp_home=stanford_path)
    if spice_score is not None:
        print(f"Overall SPICE score: {spice_score:.4f}")
        spice_f_scores = [instance['All']['f'] for instance in spice_scores_per_instance]
    else:
        spice_f_scores = [0.0] * len(candidates)

    # Use zip to avoid index errors if the data is shorter than 200
    for row, gemma_response, cider, spice, cos_sim in zip(
            qwen_data, gemma_responses, cider_scores_per_instance, spice_f_scores, cos_sim_st_per_instance_python):
        new_row = row + [
            gemma_response,
            float(cider),
            float(spice),
            float(cos_sim)
        ]
        new_table.add_data(*new_row)

    run.log({"evaluation_results": new_table})
    run.finish()