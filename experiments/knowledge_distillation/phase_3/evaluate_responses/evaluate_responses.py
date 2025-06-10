import re
import json
import wandb
import numpy as np

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


wandb.login()
run = wandb.init(project="Model Performance Comparison", name="Standard Metrics-Exp-2", entity="vlm-research")
artifact = wandb.use_artifact("vlm-research/Gemma 3 4B Distillation Phase 2/run-o9zbcvrv-VQAComparisonTable:v0")
table = artifact.get("VQA Comparison Table")
new_columns = table.columns + ["cider_score", "spice_score", "cosine_similarity"]
new_table = wandb.Table(columns=new_columns)


def extract_json(response_str):
    # Match the JSON part using regex (assumes the JSON starts with '{' and ends with '}')
    match = re.search(r'\{.*\}', response_str, re.DOTALL)
    if match:
        json_part = match.group(0)
        try:
            parsed = json.loads(json_part)
            return parsed.get("report")
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
    else:
        print("No JSON found.")
    return None


def format_for_pycocoevalcap(candidates, references_lists):
    """
    candidates: list of strings (model outputs)
    references_lists: list of lists of strings (ground truths)
                      e.g., [["ref1a", "ref1b"], ["ref2a", "ref2b", "ref2c"], ...]
    """
    gts = {}  # Ground truths
    res = {}  # Results (candidates)

    if len(candidates) != len(references_lists):
        raise ValueError("Number of candidates and reference sets must be equal.")

    for i in range(len(candidates)):
        # For pycocoevalcap, each "image" (or sample_id here) needs unique entries
        # in gts and res. The keys must match.
        sample_id = str(i) # Using index as a simple unique ID

        gts[sample_id] = []
        for ref_text in references_lists[i]:
            gts[sample_id].append({"caption": ref_text}) # Official format uses "caption"

        res[sample_id] = [{"caption": candidates[i]}]

    return gts, res


def calculate_cider(candidates, references_lists):
    """
    Calculates CIDER score.
    candidates: list of strings
    references_lists: list of lists of strings
    """
    gts, res = format_for_pycocoevalcap(candidates, references_lists)
    scorer = Cider()

    gts_cider = {k: [item['caption'] for item in v] for k, v in gts.items()}
    res_cider = {k: [item['caption'] for item in v] for k, v in res.items()}

    score, scores_per_instance = scorer.compute_score(gts_cider, res_cider)
    return score, scores_per_instance


def calculate_cosine_similarity_st(candidates, references_lists, model_name='all-MiniLM-L6-v2'):
    """
    Calculates Cosine Similarity using Sentence Transformers.
    Compares each candidate to each of its references and averages (or takes max).
    """
    model = SentenceTransformer(model_name)
    similarities = []

    for cand, refs in zip(candidates, references_lists):
        if not refs:
            similarities.append(0)
            continue

        cand_embedding = model.encode(cand, convert_to_tensor=True)
        ref_embeddings = model.encode(refs, convert_to_tensor=True)

        # Compute cosine similarity
        # util.pytorch_cos_sim returns a tensor of tensors
        cos_scores = util.pytorch_cos_sim(cand_embedding, ref_embeddings)[0]
        cos_scores_numpy = cos_scores.cpu().numpy() # Convert to numpy array

        similarities.append(np.mean(cos_scores_numpy)) # Average similarity
        # Or: similarities.append(np.max(cos_scores_numpy)) # Max similarity

    overall_avg_similarity = np.mean(similarities) if similarities else 0
    return overall_avg_similarity, similarities


def calculate_spice(candidates, references_lists, stanford_corenlp_home=None):
    """
    Calculates SPICE score.
    candidates: list of strings
    references_lists: list of lists of strings
    stanford_corenlp_home: Path to your Stanford CoreNLP directory.
                           If None, tries to use environment variables.
    """
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


if __name__ == "__main__":
    model_reports = []
    teacher_reports = []

    for row in table.data:
        try:
            model_response = row[table.columns.index("finetuned_prediction")]
            teacher_response = row[table.columns.index("teacher_prediction")]

            model_report = extract_json(model_response)
            teacher_report = extract_json(teacher_response)

            model_reports.append(model_report)
            teacher_reports.append([teacher_report])  

        except Exception as e:
            print(f"Error processing row: {e}")
            model_reports.append("")
            teacher_reports.append([""])
        

    print("--- Calculating CIDER ---")
    cider_score, cider_scores_per_instance = calculate_cider(model_reports, teacher_reports)
    if cider_score is not None:
        print(f"Overall CIDER score: {cider_score:.4f}")
        print(f"CIDER scores per instance: {cider_scores_per_instance}")

    print("\n--- Calculating Cosine Similarity (Sentence Transformers) ---")
    # This will download the model the first time it's run
    avg_cos_sim_st, cos_sim_st_per_instance = calculate_cosine_similarity_st(model_reports, teacher_reports)
    print(f"Overall Average Cosine Similarity (Sentence Transformers): {avg_cos_sim_st:.4f}")
    cos_sim_st_per_instance_python = [float(x) for x in cos_sim_st_per_instance]
    print(f"Sentence Transformer Cosine Similarities per instance (avg over refs): {cos_sim_st_per_instance_python}")

    print("\n--- Calculating SPICE ---")
    stanford_path = "/workspace/nlp_tools/stanford-corenlp-4.5.9"
    if not stanford_path:
         print("STANFORD_CORENLP_HOME not set. SPICE may fail if jars aren't found.")
         print("You can set it like: export STANFORD_CORENLP_HOME=/path/to/stanford-corenlp-full-2018-10-05")
    spice_score, spice_scores_per_instance = calculate_spice(model_reports, teacher_reports, stanford_corenlp_home=stanford_path)
    if spice_score is not None:
        print(f"Overall SPICE score: {spice_score:.4f}")
        spice_f_scores = [instance['All']['f'] for instance in spice_scores_per_instance]
        print(spice_f_scores)


    # Create new table with added score columns
    for i, row in enumerate(table.data):
        new_row = row + [
            float(cider_scores_per_instance[i]),
            float(spice_f_scores[i]),
            float(cos_sim_st_per_instance_python[i])
        ]
        new_table.add_data(*new_row)

    # Save results
    run.log({"evaluation_results": new_table})
    run.finish()