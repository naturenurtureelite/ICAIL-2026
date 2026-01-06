import torch
from transformers import pipeline

# Model ID for Llama 3.2 3B Instruct
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Initialize the pipeline
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def run_local_llama_summary(text):
    messages = [
        {"role": "system", "content": "You are a professional legal assistant."},
        {"role": "user", "content": f"Generate an extractive summary from the legal document - {text}. Choose the most important sentences."}
    ]
    
    # Llama 3.2 uses a specific prompt template
    outputs = pipe(messages, max_new_tokens=512, temperature=0.1)
    return outputs[0]["generated_text"][-1]["content"]

# Note: You must accept the model license on Hugging Face to access this.
