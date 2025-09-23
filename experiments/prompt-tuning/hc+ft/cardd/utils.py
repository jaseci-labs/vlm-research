from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from transformers import TextIteratorStreamer
import threading
from sentence_transformers import SentenceTransformer, util
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice   import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import pandas as pd
import time
import torch
import os

def get_similarity_score(reference_captions, generated_caption):
    try:
        total_score = 0.0
        for caption in reference_captions:
            ref_embed = scorer.encode(caption, convert_to_tensor=True)
            gen_embed = scorer.encode(generated_caption, convert_to_tensor=True)
            score = util.cos_sim(gen_embed, ref_embed).item()
            total_score += score

        avg_score = total_score / len(reference_captions) if reference_captions else 0.0
        return avg_score
        
    except Exception as e:
        return 0.0

def  calculate_spice(gts, res, stanford_corenlp_home=None):
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
    # --- Load model ---
    if LOAD_FROM_HF:
        print(f"🔄 Loading base model '{BASE_MODEL}'...")
        model, tokenizer = FastVisionModel.from_pretrained(
            BASE_MODEL,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )

        print(f"🔄 Applying adapter from '{MODEL_DIR}'...")
        model.load_adapter(MODEL_DIR)
    else:
        print(f"🔄 Loading full model directly from '{MODEL_DIR}'...")
        model, tokenizer = FastVisionModel.from_pretrained(
            MODEL_DIR,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )

    model.eval()
    print("✅ Model loaded successfully.")


    scorer = SentenceTransformer("all-MiniLM-L6-v2").to("cuda")
    print("✅ Sentence transformer loaded.")

    print("🚀 Starting batch evaluation...")
    all_results = {}
    cosine_scores = {}
    Spice_scores = {}
    Inference_time = {}
    Vram_usages = {}
    
    for index in indexes: 
        print(f"\n📦 Evaluating sample {index+1}/{len(indexes)} at index {index}...")
        sample = val_data[index]
        if multiple_refs:
            reference_list = sample['caption'] 
            pred, inference_time, peak_vram = run_inference(sample['image'], model, tokenizer, prompt)
            cos_score = get_similarity_score(reference_list, pred)

        else:
            reference_list = [sample['caption']]
            pred, inference_time, peak_vram = run_inference(sample['image'], model, tokenizer, prompt)
            cos_score = get_similarity_score(reference_list, pred)

        all_results[index] = pred
        cosine_scores[index] = cos_score
        Inference_time[index] = inference_time
        Vram_usages[index] = peak_vram
    gts = {}
    res = {}
    for i in range(len(indexes)):
        gts[str(i)] = [{"caption": ref} for ref in reference_list]
        res[str(i)] = [{"caption": all_results[indexes[i]]}]
    spice_score, spice_scores_per_instance = calculate_spice(gts, res)
    for i, idx in enumerate(indexes):
        Spice_scores[idx] = spice_scores_per_instance[i] if spice_scores_per_instance else 0.0

    
    print("✅ Batch evaluation complete!")
    return all_results,cosine_scores, Spice_scores, Inference_time, Vram_usages

