import torch
from transformers import pipeline

# Model ID (Llama-3.2-3B is highly recommended for multi-step reasoning)
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Initialize pipeline with optimized data types
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def run_llama_step(prompt, temperature=0.1):
    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(
        messages,
        max_new_tokens=512,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
    )
    return outputs[0]["generated_text"][-1]["content"]

def extractive_abstractive_pipeline(document):
    # --- STAGE 1: EXTRACTIVE ---
    # We use temperature 0.0 to ensure literal extraction
    print("Stage 1: Extracting key legal sentences...")
    extract_prompt = (
        f"Generate an extractive summary of the following document. "
        f"Select only verbatim sentences that represent the core facts and ruling. "
        f"Document:\n{document}"
    )
    extractive_output = run_llama_step(extract_prompt, temperature=0.0)

    # --- STAGE 2: ABSTRACTIVE ---
    # We use a slightly higher temperature for fluent rephrasing
    print("Stage 2: Synthesizing into abstractive narrative...")
    abstract_prompt = (
        f"Using the following extracted sentences as your factual base: \n{extractive_output}\n\n"
        f"Rewrite this into a single, coherent, and highly-condensed abstractive summary. "
        f"The summary must be written in a formal legal style and flow as a single narrative."
    )
    final_summary = run_llama_step(abstract_prompt, temperature=0.3)
    
    return final_summary

# Usage
legal_doc = "[Paste your document here]"
result = extractive_abstractive_pipeline(legal_doc)
print("\n--- FINAL SUMMARY ---\n", result)
