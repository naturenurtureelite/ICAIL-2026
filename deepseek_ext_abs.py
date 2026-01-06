import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Using the distilled version for local execution
model_id = "deepseek-ai/DeepSeek-R1"

# 1. Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. Setup Inference Pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def run_deepseek_task(prompt, temperature=0.6):
    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(
        messages, 
        max_new_tokens=1024, 
        temperature=temperature,
        do_sample=True if temperature > 0 else False
    )
    # Extract response and clean reasoning tags
    full_text = outputs[0]["generated_text"][-1]["content"]
    clean_text = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL).strip()
    return clean_text

def extractive_abstractive_deepseek(document):
    # --- STAGE 1: EXTRACTIVE ---
    print("Stage 1: Extracting key sentences...")
    s1_prompt = f"Extract only the most important verbatim sentences from this document:\n\n{document}"
    extractive_base = run_deepseek_task(s1_prompt, temperature=0.1)

    # --- STAGE 2: ABSTRACTIVE ---
    print("Stage 2: Synthesizing final summary...")
    s2_prompt = f"""Using these extracted sentences as a base:
    {extractive_base}
    
    Rewrite them into a single, coherent, and highly-readable abstractive summary 
    that flows naturally while preserving the original meaning."""
    
    return run_deepseek_task(s2_prompt, temperature=0.6)

# Example Usage
legal_text = "[Paste Legal Document]"
final_summary = extractive_abstractive_deepseek(legal_text)
print("\nFinal Two-Stage Summary:\n", final_summary)
