from google import genai
from google.genai import types

# Initialize the Gemini client
# Get your API key from Google AI Studio (aistudio.google.com)
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
MODEL_ID = "gemini-1.5-pro"

def run_gemini_stage(prompt, history=None):
    """Helper to call Gemini with optional chat history for context."""
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1, # Keep legal tasks deterministic
        )
    )
    return response.text

def legal_pipeline(legal_document):
    # --- STAGE 1: RHETORICAL DECOMPOSITION ---
    print("Executing Stage 1...")
    s1_prompt = f"""You are an expert legal summarizer. Decompose this legal judgment 
    into: Facts, Issues Presented, Holding/Ruling, and Court’s Reasoning. 
    Only extract salient sentences from the original text. Do not generate new text.
    
    DOCUMENT:
    {legal_document}"""
    extracted_segments = run_gemini_stage(s1_prompt)

    # --- STAGE 2: VALIDATION & CONFIDENCE SCORING ---
    print("Executing Stage 2...")
    s2_prompt = f"""Evaluate these extracted snippets. Assign a confidence score (1-5) 
    for Facts, Issues, Holding, and Reasoning based on Completeness and Fidelity.
    
    EXTRACTED SNIPPETS:
    {extracted_segments}"""
    validated_output = run_gemini_stage(s2_prompt)

    # --- STAGE 3: FLUID ABSTRACTIVE SYNTHESIS ---
    print("Executing Stage 3...")
    s3_prompt = f"""Using only these validated snippets, generate a single, coherent, 
    highly-condensed abstractive summary in a formal legal style. Integrate 
    Facts, Issue, Holding, and Reasoning into a fluid narrative.
    
    VALIDATED SNIPPETS:
    {validated_output}"""
    
    return run_gemini_stage(s3_prompt)

# Example Usage
document = "[Paste your full legal text here]"
summary = legal_pipeline(document)
print("\nFinal Legal Narrative:\n", summary)
