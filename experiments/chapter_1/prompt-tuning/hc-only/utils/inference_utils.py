from unsloth import FastVisionModel
import threading
from sentence_transformers import SentenceTransformer, util
from transformers import TextIteratorStreamer


print("🔄 Loading vision-language model...")
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)
model.eval()
print("✅ Model loaded successfully.")


print("🔄 Loading sentence transformer for scoring...")
scorer = SentenceTransformer("all-MiniLM-L6-v2").to("cuda")
print("✅ Sentence transformer loaded.")

def get_similarity_score(reference_caption, generated_caption):
    print(f"📏 Scoring similarity between:\n - Reference: {reference_caption}\n - Generated: {generated_caption}")
    try:
        ref_embed = scorer.encode(reference_caption, convert_to_tensor=True)
        gen_embed = scorer.encode(generated_caption, convert_to_tensor=True)
        score = util.cos_sim(gen_embed, ref_embed).item()
        print(f"✅ Similarity score: {score:.4f}")
        return score
    except Exception as e:
        print(f"❌ Error during similarity scoring: {e}")
        return 0.0

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
        print(f"📝 Tokenized prompt: {input_text[:100]}...")  # show a short preview

        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        inputs.pop("token_type_ids", None)  # Pixtral models don’t need this

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generated_caption = ""

        print("🚀 Starting generation thread...")
        thread = threading.Thread(
            target=model.generate,
            kwargs={
                **inputs,
                "streamer": streamer,
                "max_new_tokens": 64,
                "use_cache": True,
                "temperature": 1.0,
                "min_p": 0.1
            }
        )
        thread.start()

        # Collect the tokens from the streamer
        for token in streamer:
            generated_caption += token

        print(f"✅ Generated caption: {generated_caption.strip()}")
        return generated_caption.strip()

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return ""

def evaluate_prompt(instruction, val_data, n):
    """
    val_data: DataFrame with ['image', 'caption'] columns,
    n : number of samples per run
    """
    print(f"🧪 Evaluating instruction: {instruction}")
    total_score = 0.0
    subset = val_data.head(n)
    num_samples = len(subset)

    for i, row in subset.iterrows():
        print(f"\n--- Processing sample {i+1}/{num_samples} ---")
        try:
            pred = run_inference(row['image'], model, tokenizer, instruction)
            score = get_similarity_score(row['caption'], pred)
            total_score += score
        except Exception as e:
            print(f"❌ Skipping sample due to error: {e}")

    avg_score = total_score / num_samples
    print(f"📊 Average similarity score for instruction: {avg_score:.4f}")
    return avg_score
