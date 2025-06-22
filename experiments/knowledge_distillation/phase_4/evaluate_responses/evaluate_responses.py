import re
import json
import wandb
import numpy as np

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from sentence_transformers import SentenceTransformer, util


wandb.login()
run = wandb.init(project="Gemma 3 4B Distillation Phase 2", name="Evaluation scores for 8-bits", entity="vlm-research")
artifact = wandb.use_artifact("vlm-research/Gemma 3 4B Distillation Phase 2/run-kt20hs7e-VQAComparisonTable-afb52717eb3942f27c14650247939ad0:v0")
table = artifact.get("VQA Comparison Table")
new_columns = table.columns + ["baseline_cider_score", "baseline_spice_score", "baseline_cosine_similarity", "distilled_cider_score", "distilled_spice_score", "distilled_cosine_similarity"]
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


def extract_base_model_response(response_str):
    """
    Extracts the main model response from a baseline_prediction string.
    Assumes the model response is the text after the first occurrence of 'model' (case-insensitive),
    or after a specific marker, or simply returns the input if no marker is found.
    """
    match = re.search(r'model\s*[\n:]*\s*(.*)', response_str, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_str.strip()


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
    import pycocoevalcap.spice.get_stanford_models as gsm
    def skip_model_download():
        print("Skipping Stanford CoreNLP download - using local files.")
    gsm.get_stanford_models = skip_model_download
    
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
    base_model_reports = []
    distilled_model_reports = []
    teacher_reports = []

    for row in table.data:
        try:
            baseline_model_response = row[table.columns.index("baseline_prediction")]
            distilled_model_response = row[table.columns.index("finetuned_prediction")]
            teacher_response = row[table.columns.index("teacher_prediction")]

            # Use extract_base_model_response for baseline_prediction
            base_model_report = extract_base_model_response(baseline_model_response)
            distilled_model_report = extract_json(distilled_model_response)
            teacher_report = extract_json(teacher_response)

            base_model_reports.append(base_model_report)
            distilled_model_reports.append(distilled_model_report)
            teacher_reports.append([teacher_report])

        except Exception as e:
            print(f"Error processing row: {e}")
            base_model_reports.append("")
            distilled_model_reports.append("")
            teacher_reports.append([""])


    # --------------------------------------------------------------------------
    # CIDEr score calculation
    # --------------------------------------------------------------------------
    print("--- Calculating CIDER ---")
    base_response_cider_score, base_response_cider_scores_per_instance = calculate_cider(base_model_reports, teacher_reports)
    if base_response_cider_score is not None:
        print(f"Overall CIDER score: {base_response_cider_score:.4f}")
        print(f"CIDER scores per instance: {base_response_cider_scores_per_instance}")

    distilled_response_cider_score, distilled_response_cider_scores_per_instance = calculate_cider(distilled_model_reports, teacher_reports)
    if distilled_response_cider_score is not None:
        print(f"Overall Distilled CIDER score: {distilled_response_cider_score:.4f}")
        print(f"Distilled CIDER scores per instance: {distilled_response_cider_scores_per_instance}")


    # ------------------------------------------------------------------------------
    # Cosine Similarity score calculation
    # ------------------------------------------------------------------------------
    print("\n--- Calculating Cosine Similarity (Sentence Transformers) ---")
    # This will download the model the first time it's run
    base_response_avg_cos_sim_st, base_response_cos_sim_st_per_instance = calculate_cosine_similarity_st(base_model_reports, teacher_reports)
    print(f"Overall Base response Average Cosine Similarity (Sentence Transformers): {base_response_avg_cos_sim_st:.4f}")
    base_response_cos_sim_st_per_instance_python = [float(x) for x in base_response_cos_sim_st_per_instance]
    print(f"Base response Sentence Transformer Cosine Similarities per instance (avg over refs): {base_response_cos_sim_st_per_instance_python}")


    distilled_response_avg_cos_sim_st, distilled_response_cos_sim_st_per_instance = calculate_cosine_similarity_st(distilled_model_reports, teacher_reports)
    print(f"Overall Distilled Average Cosine Similarity (Sentence Transformers): {distilled_response_avg_cos_sim_st:.4f}")
    distilled_response_cos_sim_st_per_instance_python = [float(x) for x in distilled_response_cos_sim_st_per_instance]
    print(f"Distilled Sentence Transformer Cosine Similarities per instance (avg over refs): {distilled_response_cos_sim_st_per_instance_python}")


    # ------------------------------------------------------------------------------
    # SPICE score calculation
    # ------------------------------------------------------------------------------
    print("\n--- Calculating SPICE ---")
    stanford_path = "/workspace/nlp_tools/stanford-corenlp-4.5.10"
    if not stanford_path:
         print("STANFORD_CORENLP_HOME not set. SPICE may fail if jars aren't found.")
         print("You can set it like: export STANFORD_CORENLP_HOME=/path/to/stanford-corenlp-full-2018-10-05")
    base_response_spice_score, base_response_spice_scores_per_instance = calculate_spice(base_model_reports, teacher_reports, stanford_corenlp_home=stanford_path)
    if base_response_spice_score is not None:
        print(f"Overall Base response SPICE score: {base_response_spice_score:.4f}")
        base_response_spice_f_scores = [instance['All']['f'] for instance in base_response_spice_scores_per_instance]
        print(base_response_spice_f_scores)

    distilled_response_spice_score, distilled_response_spice_scores_per_instance = calculate_spice(distilled_model_reports, teacher_reports, stanford_corenlp_home=stanford_path)
    if distilled_response_spice_score is not None:
        print(f"Overall Distilled response SPICE score: {distilled_response_spice_score:.4f}")
        distilled_response_spice_f_scores = [instance['All']['f'] for instance in distilled_response_spice_scores_per_instance]
        print(distilled_response_spice_f_scores)


    # ------------------------------------------------------------------------------
    # Create new table with added score columns
    # ------------------------------------------------------------------------------
    for i, row in enumerate(table.data):
        new_row = row + [
            float(base_response_cider_scores_per_instance[i]),
            float(base_response_spice_f_scores[i]),
            float(base_response_cos_sim_st_per_instance_python[i]),
            float(distilled_response_cider_scores_per_instance[i]),
            float(distilled_response_spice_f_scores[i]),
            float(distilled_response_cos_sim_st_per_instance_python[i])
        ]
        new_table.add_data(*new_row)


    # ------------------------------------------------------------------------------
    # Print Final scores
    # ------------------------------------------------------------------------------
    print("\n--- Evaluation Results ---")
    print("Base CIDEr score: ", base_response_cider_score)
    print("Base SPICE score: ", base_response_spice_score)
    print("Base Cosine Similarity (Sentence Transformers): ", base_response_avg_cos_sim_st)

    print("Distilled CIDEr score: ", distilled_response_cider_score)
    print("Distilled SPICE score: ", distilled_response_spice_score)
    print("Distilled Cosine Similarity (Sentence Transformers): ", distilled_response_avg_cos_sim_st)


    # Save results
    run.log({"evaluation_results": new_table})
    run.finish()