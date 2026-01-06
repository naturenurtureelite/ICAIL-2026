import openai
from openai import OpenAI

# Initialize the client
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

def generate_abstractive_summary(legal_text):
    """
    Generates a coherent, natural-language summary that preserves 
    legal nuance and core details.
    """
    
    user_prompt = f"""Generate an abstractive summary from the legal document - 
    {legal_text}. Generate the information in the summary in a readable, 
    coherent and natural manner while preserving the core meaning and 
    important details."""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a legal expert assistant. Your task is to synthesize complex legal language into clear, professional, and natural prose without losing the document's legal weight."
                },
                {"role": "user", "content": user_prompt}
            ],
            # A low temperature (0.2-0.3) is best for abstractive 
            # legal work—it allows for fluidity without being 'creative'.
            temperature=0.2 
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {e}"

# Example Usage
document_content = """[Paste Legal Document Content Here]"""
print(generate_abstractive_summary(document_content))
