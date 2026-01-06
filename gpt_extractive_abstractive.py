import openai
from openai import OpenAI

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

def two_stage_legal_summary(legal_document):
    # --- STAGE 1: EXTRACTIVE ---
    print("Executing Stage 1: Extraction...")
    
    stage1_prompt = f"""Generate an extractive summary from the legal document - 
    {legal_document}. Choose the most important sentences from the legal 
    document to form the legal document summary."""

    extraction_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a precise legal clerk. Extract key sentences exactly as they appear."},
            {"role": "user", "content": stage1_prompt}
        ],
        temperature=0
    )
    extractive_summary = extraction_response.choices[0].message.content

    # --- STAGE 2: ABSTRACTIVE REPHRASING ---
    print("Executing Stage 2: Abstractive Refinement...")
    
    stage2_prompt = f"""Given the legal document- {legal_document} and a summary 
    containing some important sentences about the legal document - {extractive_summary}, 
    rephrase the information in the summary in a more readable, coherent and 
    natural manner while preserving the core meaning and important details."""

    final_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a legal communications expert. Refine the provided extraction into a professional, readable narrative."},
            {"role": "user", "content": stage2_prompt}
        ],
        temperature=0.3 # Slightly higher for natural flow
    )

    return final_response.choices[0].message.content

# Example Execution
doc = """[Paste Legal Document Content Here]"""
final_summary = two_stage_legal_summary(doc)
print("\nFinal Coherent Summary:\n", final_summary)
