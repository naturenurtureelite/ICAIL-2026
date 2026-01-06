import torch
from transformers import pipeline

# Model ID: Using the 3B Instruct version for better reasoning/synthesis
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Initialize the pipeline
# Note: Use bfloat16 for efficiency if your GPU supports it
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def generate_abstractive_summary(legal_document):
    # Llama 3.2 uses a specific Chat Template format
    messages = [
        {
            "role": "system", 
            "content": "You are a professional legal assistant. Your goal is to rewrite complex documents into clear, natural-sounding summaries while preserving all critical legal meanings."
        },
        {
            "role": "user", 
            "content": f"Generate an abstractive summary from the following legal document. Ensure the summary is readable, coherent, and preserves the core details:\n\n{legal_document}"
        }
    ]

    # Generation parameters
    # Abstractive tasks benefit from a slightly higher temperature (0.3-0.6) 
    # to allow the model to rephrase fluently.
    outputs = pipe(
        messages,
        max_new_tokens=512,
        temperature=0.3, 
        do_sample=True,
    )

    return outputs[0]["generated_text"][-1]["content"]

# Example usage
doc_text = "[Your Legal Document Text Here]"
print(generate_abstractive_summary(doc_text))
