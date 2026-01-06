from google import genai
from google.genai import types

# Initialize the client with your API key
# You can get one for free at https://aistudio.google.com/
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

def generate_extractive_summary(legal_document_text):
    prompt_content = f"""Generate an extractive summary from the legal document - 
    {legal_document_text}. Choose the most important sentences 
    from the legal document to form the legal document summary."""

    try:
        # Using Gemini 1.5 Pro for high-fidelity legal extraction
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt_content,
            config=types.GenerateContentConfig(
                # Low temperature ensures the model stays literal 
                # and picks existing sentences rather than paraphrasing.
                temperature=0.0 
            )
        )
        
        return response.text

    except Exception as e:
        return f"An error occurred: {e}"

# Example Usage:
document = """[Paste your long legal text here]"""
extractive_summary = generate_extractive_summary(document)
print(extractive_summary)
