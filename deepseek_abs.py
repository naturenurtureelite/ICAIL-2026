import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Using the distilled version for local execution
model_id = "deepseek-ai/DeepSeek-R1"

# 1. Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. Setup Pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def deepseek_abstractive_summary(text):
    # Prompt for synthesis and natural rephrasing
    # DeepSeek recommends NO system prompt for R1 models.
    prompt = f"""Generate an abstractive summary of the following document. 
    Rewrite the core information in a readable, coherent, and natural manner 
    while preserving all important details and legal nuances.
    
    DOCUMENT:
    {text}
    
    SUMMARY:"""

    messages = [{"role": "user", "content": prompt}]
    
    # 0.6 is the recommended temperature for R1 reasoning and synthesis
    outputs = pipe(
        messages, 
        max_new_tokens=1024, 
        temperature=0.6,
        do_sample=True
    )
    
    full_response = outputs[0]["generated_text"][-1]["content"]
    
    # Optional: Logic to separate reasoning from the final answer
    # Most R1 users prefer to strip the <think> block for the final user view.
    clean_summary = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
    
    return clean_summary

# Example Usage
document = "[Paste your text here]"
print(deepseek_abstractive_summary(document))
