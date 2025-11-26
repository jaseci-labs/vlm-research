from unsloth import FastVisionModel
from transformers import TextIteratorStreamer
import threading
from sentence_transformers import SentenceTransformer, util
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import pandas as pd
from cider import Cider
import time
import torch

PICKLE_PATH = "/workspace/flickr-df.p"

# --- Load model ---
print("🔄 Loading vision-language model...")
model, tokenizer = FastVisionModel.from_pretrained(
    "/workspace/flickr_qw_finetune",           #replace with your model path
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
model.eval()
print("✅ Model loaded successfully.")

# --- Load scorer ---
print("🔄 Loading sentence transformer for scoring...")
scorer = SentenceTransformer("all-MiniLM-L6-v2").to("cuda")
print("✅ Sentence transformer loaded.")

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

def score_per_image(refs, hypos):
    scorers = [
        (Meteor(), "METEOR"),
        (Spice(), "SPICE")
    ]
    ptb = PTBTokenizer()
    refs_wrapped = {i: [{"caption": c} for c in caps] for i, caps in refs.items()}
    hypos_wrapped = {i: [{"caption": hypos[i][0]}] for i in hypos}
    refs_tok = ptb.tokenize(refs_wrapped)
    hypos_tok = ptb.tokenize(hypos_wrapped)
    all_scores = {}
    for scorer, name in scorers:
        avg_score, per_image_scores = scorer.compute_score(refs_tok, hypos_tok)
        for idx, img_id in enumerate(hypos_tok.keys()):
            all_scores.setdefault(img_id, {})
            if name == "SPICE":
                f_all = per_image_scores[idx].get("All", {}).get("f", 0.0)
                all_scores[img_id][name] = f_all
            else:
                all_scores[img_id][name] = per_image_scores[idx]
    return all_scores

def evaluate_cider(hypos, refs):
    gts = {str(i): refs[i] for i in refs}
    res = [{"image_id": str(i), "caption": [hypos[i][0] if isinstance(hypos[i], list) else hypos[i]]} for i in hypos]
    cider = Cider()
    score, individual_scores = cider.compute_score(gts, res, PICKLE_PATH)
    return score, individual_scores

def run_inference(image, model, tokenizer, instruction):
    print(f"🧠 Running inference with instruction: {instruction}")
    try:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        print(f"📝 Tokenized prompt: {input_text[:100]}...")
        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        inputs.pop("token_type_ids", None)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generated_caption = ""
        print("🚀 Starting generation thread...")
        # Measure generation time and VRAM
        gen_start_time = time.time()
        torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats
        vram_reserved_before = torch.cuda.memory_reserved() / 1024**3
        vram_allocated_before = torch.cuda.memory_allocated() / 1024**3
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
        for token in streamer:
            generated_caption += token
        thread.join()
        vram_reserved_after = torch.cuda.memory_reserved() / 1024**3
        vram_allocated_after = torch.cuda.max_memory_allocated() / 1024**3  # Use peak allocated memory
        gen_end_time = time.time()
        generation_time = gen_end_time - gen_start_time
        vram_reserved = max(0, vram_reserved_after - vram_reserved_before)
        vram_allocated = max(0, vram_allocated_after - vram_allocated_before)
        print(f"✅ Generated caption: {generated_caption.strip()}")
        print(f"🔹 Generation time: {generation_time:.2f} seconds")
        print(f"🔹 VRAM Reserved (End - Start): {vram_reserved:.2f} GB")
        print(f"🔹 VRAM Allocated (Peak - Start): {vram_allocated:.2f} GB")
        return generated_caption.strip(), generation_time, vram_reserved, vram_allocated
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return "", 0.0, 0.0, 0.0

def evaluate_sample(prompts, sample, multiple_refs):
    print(f"\n🔍 Starting evaluation for sample with reference: {sample['caption']}")
    hypos = dict()
    cosine_scores = []
    inference_times = []
    vram_reserved_list = []
    vram_allocated_list = []
    if multiple_refs:
        ref_cap_list = sample['caption']
    else:
        ref_cap_list = [sample['caption']]
    refs = {i: ref_cap_list for i in range(len(prompts))}
    for i, prompt in enumerate(prompts):
        print(f"🧪 Evaluating instruction {i+1}/{len(prompts)}: '{prompt}'")
        pred, gen_time, vram_reserved, vram_allocated = run_inference(sample['image'], model, tokenizer, prompt)
        print(f"🔹 Generated: {pred}")
        print(f"🔹 Generation time: {gen_time:.2f} seconds")
        cos_score = get_similarity_score(ref_cap_list, pred)
        print(f"🔹 Semantic similarity: {cos_score:.4f}")
        cosine_scores.append(cos_score)
        inference_times.append(gen_time)
        vram_reserved_list.append(vram_reserved)
        vram_allocated_list.append(vram_allocated)
        hypos[i] = [pred]
    print("📊 Scoring predictions with COCO metrics...")
    coco_scores = score_per_image(refs, hypos)
    _, cider_scores = evaluate_cider(hypos, refs)
    results = []
    for i, prompt in enumerate(prompts):
        res = {
            "reference_captions": " || ".join(ref_cap_list),
            "generated": hypos[i][0] if isinstance(hypos[i], list) else hypos[i],
            "semantic_similarity": cosine_scores[i],
            "METEOR": coco_scores[i].get("METEOR", 0.0),
            "CIDEr": float(cider_scores[i]) if cider_scores[i] is not None else 0.0,
            "SPICE": coco_scores[i].get("SPICE", 0.0),
            "inference_time": inference_times[i],
            "vram_reserved_gb": vram_reserved_list[i],
            "vram_allocated_gb": vram_allocated_list[i]
        }
        print(f"✅ Result for instruction {i+1}: {res}")
        results.append(res)
    return results

def evaluate_batch(prompts_list, val_data, indexes, multiple_refs=True):
    print("🚀 Starting batch evaluation...")
    all_results = []
    for i, (index, prompts) in enumerate(zip(indexes, prompts_list)):
        print(f"\n📦 Evaluating sample {i+1}/{len(indexes)} at index {index}...")
        if multiple_refs:
            results = evaluate_sample(prompts, val_data[index], multiple_refs)
        else:
            results = evaluate_sample(prompts, val_data.loc[index], multiple_refs)
        for r in results:
            r["sample_index"] = index
        all_results.append(results)
    print("\n🔄 Transposing results by prompt...")
    transposed = list(map(list, zip(*all_results)))
    print(f"📁 Creating {len(transposed)} DataFrames (one per prompt)...")
    dfs = [pd.DataFrame(rows) for rows in transposed]
    print("✅ Batch evaluation complete!")
    return dfs