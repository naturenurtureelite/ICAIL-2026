from google import genai
from google.genai import types

# Initialize the Gemini client
# Get your API key from https://aistudio.google.com/
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

def generate_two_stage_summary(legal_document_text):
    # --- STAGE 1: EXTRACTIVE SUMMARY ---
    # Goal: Isolate the most important verbatim sentences.
    print("Executing Stage 1: Extraction...")
    prompt_stage1 = f"""Generate an extractive summary from the legal document - 
    {legal_document_text}. Choose the most important sentences 
    from the legal document to form the legal document summary."""

    response_s1 = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=prompt_stage1,
        config=types.GenerateContentConfig(temperature=0.0) # Maximum literalness
    )
    extractive_summary = response_s1.text

    # --- STAGE 2: ABSTRACTIVE REFINEMENT ---
    # Goal: Synthesize the extracted facts into a natural, fluid narrative.
    print("Executing Stage 2: Abstractive Synthesis...")
    prompt_stage2 = f"""Given the legal document- {legal_document_text} and a summary 
    containing some important sentences about the legal document - {extractive_summary}, 
    rephrase the information in the summary in a more readable, coherent and 
    natural manner while preserving the core meaning and important details."""

    response_s2 = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=prompt_stage2,
        config=types.GenerateContentConfig(temperature=0.3) # Slightly more fluid
    )
    
    return response_s2.text

# Example usage
legal_text = """[Paste your long legal text here]"""
final_summary = generate_two_stage_summary(legal_text)
print("\n--- FINAL SUMMARY ---\n", final_summary)
