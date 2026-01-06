import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Using the distilled 8B version (based on Llama) for standard hardware
# Full R1 (671B) would require the deepseek-ai/DeepSeek-R1 ID and 800GB+ VRAM
model_id = "deepseek-ai/DeepSeek-R1"

# 1. Load Tokenizer and Model
# Note: DeepSeek-R1 works best with bfloat16 or float16 precision
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 2. Create the Pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def deepseek_extractive_summary(text):
    # Prompt engineering for DeepSeek-R1:
    # We explicitly tell it NOT to rewrite, only to extract.
    # Note: R1 usually outputs a <think> block first.
    prompt = f"""Extract the most critical verbatim sentences from the following document. 
    Do not paraphrase. Do not add external information. 
    
    DOCUMENT:
    {text}
    
    EXTRACTED SENTENCES:"""

    messages = [{"role": "user", "content": prompt}]
    
    # Generation Settings
    # temperature=0.6 is recommended by DeepSeek for R1 to ensure logical reasoning
    # but for strict extraction, 0.1 is safer to prevent hallucination.
    outputs = pipe(
        messages, 
        max_new_tokens=1024, 
        temperature=0.1,
        do_sample=False # Set to False for deterministic extraction
    )
    
    return outputs[0]["generated_text"][-1]["content"]

# Example Usage
document = "[Paste your legal or technical text here]"
print(deepseek_extractive_summary(document))
