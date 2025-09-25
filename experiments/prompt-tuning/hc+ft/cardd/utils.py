from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from transformers import TextIteratorStreamer
import threading
from sentence_transformers import SentenceTransformer, util
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice   import Spice
from pycocoevalcap.cider.cider   import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import pandas as pd
import time
import torch
import os

def get_similarity_score(reference_captions, generated_caption, scorer):
    try:
        if not reference_captions or not generated_caption:
            return 0.0

        total_score = 0.0
        for caption in reference_captions:
            ref_embed = scorer.encode(str(caption), convert_to_tensor=True)
            gen_embed = scorer.encode(str(generated_caption), convert_to_tensor=True)
            score = util.cos_sim(gen_embed, ref_embed).item()
            total_score += score

        avg_score = total_score / len(reference_captions) if reference_captions else 0.0
        return avg_score

    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0.0

def calculate_spice(gts, res, stanford_corenlp_home=None):
    """
    Calculates SPICE score.
    candidates: indexed dict of {str: list of dicts with 'caption' key}
    references_lists: matching indexed dict of {str: list of dicts with 'caption' key}
    stanford_corenlp_home: Path to your Stanford CoreNLP directory.
                           If None, tries to use environment variables.
    """

    gts_spice = {k: [item['caption'] for item in v] for k, v in gts.items()}
    res_spice = {k: [item['caption'] for item in v] for k, v in res.items()}

    if stanford_corenlp_home:
        print(f"Temporarily set STANFORD_CORENLP_HOME to: {stanford_corenlp_home}")

    scorer = Spice()
    try:
        score, scores_per_instance = scorer.compute_score(gts_spice, res_spice)
        spice_f_scores = [instance['All']['f'] for instance in scores_per_instance]

    except Exception as e:
        print(f"Error calculating SPICE: {e}")
        print("Ensure Stanford CoreNLP is correctly set up (jars accessible, sufficient memory).")
        print("Try setting SPICE_JAR and STANFORD_CORENLP_MODELS_JAR environment variables.")
        print("Or, place stanford-corenlp-X.X.X.jar and stanford-corenlp-X.X.X-models.jar")
        print("in pycocoevalcap/spice/lib/ (you might need to create this path).")
        return None, None
    return score, spice_f_scores

def calculate_cider(gts, res):
    """
    Calculates CIDER score.
    candidates: indexed dict of {str: list of dicts with 'caption' key}
    references_lists: matching indexed dict of {str: list of dicts with 'caption' key}
    """

    gts_cider = {k: [item['caption'] for item in v] for k, v in gts.items()}
    res_cider = {k: [item['caption'] for item in v] for k, v in res.items()}

    scorer = Cider()
    try:
        score, scores_per_instance = scorer.compute_score(gts_cider, res_cider)
        return score, scores_per_instance

    except Exception as e:
        print(f"Error calculating CIDER: {e}")
        return None, None

def run_inference(image, model, tokenizer, instruction):
    """
    Runs inference on `image` + `instruction` through `model`/`tokenizer`,
    and returns:
      - generated_caption (str)
      - inference_time_s (float)
      - peak_vram_mb (float)
    """
    try:
        # Build the chat prompt
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        print(f"📝 Tokenized prompt: {input_text[:100]}...")

        # Prepare inputs and move to CUDA
        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        inputs.pop("token_type_ids", None)

        # Reset CUDA memory stats so we can track this run’s peak
        torch.cuda.reset_peak_memory_stats(device="cuda")

        # Start timing
        t_start = time.time()

        # Set up the streamer and launch generation in a thread
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = threading.Thread(
            target=model.generate,
            kwargs={
                **inputs,
                "streamer": streamer,
                "max_new_tokens": 128,
                "use_cache": True,
                "temperature": 1.0,
                "min_p": 0.1
            }
        )
        thread.start()

        # Collect tokens
        generated_caption = ""
        for token in streamer:
            generated_caption += token

        # Ensure generation is done
        thread.join()

        # Stop timing
        t_end = time.time()
        inference_time_s = t_end - t_start

        # Peak VRAM usage in bytes → convert to MiB
        peak_vram_bytes = torch.cuda.max_memory_allocated(device="cuda")
        peak_vram_mb = peak_vram_bytes / (1024 ** 2)

        return generated_caption.strip(), inference_time_s, peak_vram_mb

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        # On error, return empty caption and zeros
        return "", 0.0, 0.0

def evaluate_batch(prompt, val_data, indexes, multiple_refs=True, MODEL_DIR="/workspace/unsloth-finetune", LOAD_FROM_HF=False):
    """
    prompts_list: list of instructions to evaluate
    val_data: DataFrame with ['image', 'caption'] columns,
    indexes: list of indexes to sample from val_data
    """
    print(f"🔄 Loading vision-language model from {MODEL_DIR}...")
    BASE_MODEL = "unsloth/Qwen2-VL-7B-Instruct"
    temp_dir = None  # Track temporary directory for cleanup

    # --- Load model ---
    try:
        if LOAD_FROM_HF:
            # For HF loading, first download the model to avoid Unsloth parsing issues
            print(f"🔄 Downloading model from Hugging Face: '{MODEL_DIR}'...")

            # Create a temporary local directory for the downloaded model
            import tempfile
            import shutil
            from huggingface_hub import snapshot_download

            temp_dir = tempfile.mkdtemp(prefix="hf_model_")
            print(f"🔄 Downloading to temporary directory: {temp_dir}")

            try:
                # Download the model
                snapshot_download(
                    repo_id=MODEL_DIR,
                    local_dir=temp_dir,
                    ignore_patterns=["*.git*", "README.md", "*.txt"]
                )

                print(f"✅ Model downloaded successfully")
                print(f"🔄 Loading model from local directory: {temp_dir}")

                # Now load from the local directory
                model, tokenizer = FastVisionModel.from_pretrained(
                    temp_dir,
                    load_in_4bit=True,
                    use_gradient_checkpointing="unsloth",
                )

            except Exception as download_error:
                print(f"❌ Error downloading model: {download_error}")
                # Cleanup and try direct loading
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                raise download_error

        else:
            # Check if MODEL_DIR exists as a local directory
            if os.path.exists(MODEL_DIR) and os.path.isdir(MODEL_DIR):
                print(f"🔄 Loading locally fine-tuned model from '{MODEL_DIR}'...")
                model, tokenizer = FastVisionModel.from_pretrained(
                    MODEL_DIR,
                    load_in_4bit=True,
                    use_gradient_checkpointing="unsloth",
                )
            else:
                # Fallback: treat MODEL_DIR as a HF repo ID and download first
                print(f"🔄 Local directory not found. Downloading from Hugging Face: '{MODEL_DIR}'...")

                import tempfile
                import shutil
                from huggingface_hub import snapshot_download

                temp_dir = tempfile.mkdtemp(prefix="hf_model_")
                print(f"🔄 Downloading to temporary directory: {temp_dir}")

                try:
                    snapshot_download(
                        repo_id=MODEL_DIR,
                        local_dir=temp_dir,
                        ignore_patterns=["*.git*", "README.md", "*.txt"]
                    )

                    model, tokenizer = FastVisionModel.from_pretrained(
                        temp_dir,
                        load_in_4bit=True,
                        use_gradient_checkpointing="unsloth",
                    )

                except Exception as download_error:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    raise download_error

    except Exception as e:
        print(f"❌ Error loading model from {MODEL_DIR}: {e}")
        print(f"🔍 LOAD_FROM_HF flag: {LOAD_FROM_HF}")
        print(f"🔍 MODEL_DIR exists locally: {os.path.exists(MODEL_DIR) if not LOAD_FROM_HF else 'N/A (loading from HF)'}")
        raise

    model.eval()
    print("✅ Model loaded successfully.")

    # Initialize sentence transformer for cosine similarity
    sentence_scorer = SentenceTransformer("all-MiniLM-L6-v2").to("cuda")
    print("✅ Sentence transformer loaded.")

    print("🚀 Starting batch evaluation...")
    all_results = {}
    all_references = {}
    cosine_scores = {}
    Spice_scores = {}
    Cider_scores = {}
    Inference_time = {}
    Vram_usages = {}

    for index in indexes:
        print(f"\n📦 Evaluating sample {index+1}/{len(indexes)} at index {index}...")
        sample = val_data[index]
        if multiple_refs:
            reference_list = sample['caption']
        else:
            reference_list = [sample['caption']]

        pred, inference_time, peak_vram = run_inference(sample['image'], model, tokenizer, prompt)
        print(f"🔍 Generated prediction: '{pred[:100]}...'" if len(pred) > 100 else f"🔍 Generated prediction: '{pred}'")
        print(f"🔍 Reference captions: {reference_list}")

        cos_score = get_similarity_score(reference_list, pred, sentence_scorer)
        print(f"🔍 Cosine similarity score: {cos_score}")

        all_results[index] = pred
        all_references[index] = reference_list
        cosine_scores[index] = cos_score
        Inference_time[index] = inference_time
        Vram_usages[index] = peak_vram

    # Prepare data for SPICE and CIDER evaluation
    gts = {}
    res = {}
    for i, idx in enumerate(indexes):
        gts[str(i)] = [{"caption": ref} for ref in all_references[idx]]
        res[str(i)] = [{"caption": all_results[idx]}]

    # Calculate SPICE scores
    spice_score, spice_scores_per_instance = calculate_spice(gts, res)
    if spice_scores_per_instance is not None and len(spice_scores_per_instance) > 0:
        for i, idx in enumerate(indexes):
            Spice_scores[idx] = spice_scores_per_instance[i]
    else:
        for idx in indexes:
            Spice_scores[idx] = 0.0

    # Calculate CIDER scores
    cider_score, cider_scores_per_instance = calculate_cider(gts, res)
    if cider_scores_per_instance is not None and len(cider_scores_per_instance) > 0:
        for i, idx in enumerate(indexes):
            Cider_scores[idx] = cider_scores_per_instance[i]
    else:
        for idx in indexes:
            Cider_scores[idx] = 0.0

    print("✅ Batch evaluation complete!")

    # Cleanup temporary directory if it was created
    if temp_dir and os.path.exists(temp_dir):
        print(f"🧹 Cleaning up temporary directory: {temp_dir}")
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print("✅ Temporary directory cleaned up")
        except Exception as cleanup_error:
            print(f"⚠️ Could not clean up temporary directory: {cleanup_error}")

    return all_results, cosine_scores, Spice_scores, Cider_scores, Inference_time, Vram_usages

