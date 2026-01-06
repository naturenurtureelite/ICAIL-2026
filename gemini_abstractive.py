from google import genai
from google.genai import types

# Initialize the client
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

def generate_abstractive_summary(legal_text):
    """
    Generates a coherent, natural-language summary that synthesizes 
    legal details into a readable format.
    """
    
    prompt = f"""Generate an abstractive summary from the legal document
    - {legal_text}. Generate the information in the
    summary in a readable, coherent and natural manner while
    preserving the core meaning and important details."""

    try:
        # We use a slightly higher temperature (e.g., 0.2 to 0.4) 
        # to allow the model the 'creativity' to rephrase and 
        # connect ideas smoothly, without risking hallucination.
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024  # Optional: Limit length
            )
        )
        
        return response.text

    except Exception as e:
        return f"Error: {e}"

# Example Usage
legal_doc = """[Insert full legal document here]"""
print(generate_abstractive_summary(legal_doc))
