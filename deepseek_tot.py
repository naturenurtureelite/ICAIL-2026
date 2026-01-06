import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Initialize model (add quantization_config if VRAM is limited)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def call_r1(prompt, temperature=0.6):
    # DeepSeek R1 performs best without a system prompt
    messages = [{"role": "user", "content": prompt}]
    out = pipe(messages, max_new_tokens=1024, temperature=temperature, do_sample=True)
    full_resp = out[0]["generated_text"][-1]["content"]
    
    # Extract the final answer and the 'thought' block separately
    thought = re.search(r'<think>(.*?)</think>', full_resp, re.DOTALL)
    answer = re.sub(r'<think>.*?</think>', '', full_resp, re.DOTALL).strip()
    return answer, (thought.group(1).strip() if thought else "")

def legal_3stage_pipeline(document):
    # --- STAGE 1: RHETORICAL DECOMPOSITION ---
    print("Stage 1: Decomposing document structure...")
    s1_prompt = f"Decompose this legal text into verbatim snippets for: Facts, Issues, and Decision.\n\nTEXT:\n{document}"
    extracted_data, _ = call_r1(s1_prompt, temperature=0.1)

    # --- STAGE 2: REASONED VALIDATION ---
    print("Stage 2: Reasoning & self-verification...")
    s2_prompt = f"Verify these snippets against the goal of legal accuracy. Identify any missing nuances.\n\nSNIPPETS:\n{extracted_data}"
    # We keep the 'thought' here because it contains the model's critique
    validation_critique, reasoning_logic = call_r1(s2_prompt, temperature=0.4)

    # --- STAGE 3: ABSTRACTIVE SYNTHESIS ---
    print("Stage 3: Final executive synthesis...")
    # We feed both the snippets and the critique into the final stage
    s3_prompt = f"""Generate a formal 1-paragraph abstractive summary. 
    Base it on these snippets: {extracted_data} 
    And these corrections: {validation_critique}"""
    final_summary, _ = call_r1(s3_prompt, temperature=0.6)
    
    return final_summary

# Usage
doc = "[Paste Document]"
summary = legal_3stage_pipeline(doc)
print(f"\n--- FINAL R1 SUMMARY ---\n{summary}")
