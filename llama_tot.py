import torch
from transformers import pipeline

# Model ID (Llama-3.2-3B is excellent for multi-stage reasoning)
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Initialize the text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def run_hf_stage(prompt, temperature=0.1):
    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(
        messages, 
        max_new_tokens=1024, 
        temperature=temperature,
        do_sample=True if temperature > 0 else False
    )
    return outputs[0]["generated_text"][-1]["content"]

def legal_judgment_pipeline(legal_doc):
    # --- STAGE 1: RHETORICAL DECOMPOSITION (EXTRACTIVE) ---
    print("Stage 1: Decomposing document into rhetorical segments...")
    s1_prompt = f"""You are an expert legal summarizer. Decompose this legal judgment 
    into its core segments: Facts, Issues, Holding, and Reasoning. 
    Only extract verbatim sentences from the text. Do not generate new text.
    
    DOCUMENT:
    {legal_doc}"""
    segments = run_hf_stage(s1_prompt, temperature=0.0)

    # --- STAGE 2: VALIDATION & CONFIDENCE SCORING ---
    print("Stage 2: Validating extracted segments...")
    s2_prompt = f"""Evaluate the following extracted snippets for Completeness and Fidelity. 
    Assign a confidence score (1-5) for each segment:
    
    EXTRACTED SNIPPETS:
    {segments}"""
    validated = run_hf_stage(s2_prompt, temperature=0.0)

    # --- STAGE 3: FLUID ABSTRACTIVE SYNTHESIS ---
    print("Stage 3: Synthesizing final narrative...")
    s3_prompt = f"""Using only the validated snippets below, generate a single, 
    coherent, and highly-condensed abstractive summary in a formal legal style. 
    Seamlessly integrate Facts, Issue, Holding, and Reasoning into a fluid narrative.
    
    VALIDATED SNIPPETS:
    {validated}"""
    final_summary = run_hf_stage(s3_prompt, temperature=0.2)
    
    return final_summary

# Usage
judgment_text = "[Paste Full Legal Text Here]"
final_result = legal_judgment_pipeline(judgment_text)
print("\n--- FINAL LEGAL SUMMARY ---\n", final_result)
