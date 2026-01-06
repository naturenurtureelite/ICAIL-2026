import openai
from openai import OpenAI

# Initialize the client with your API key
client = OpenAI(api_key="your-api-key-here")

def generate_legal_summary(legal_document_text):
    prompt_content = f"""Generate an extractive summary from the legal document - 
    {legal_document_text}. Choose the most important sentences 
    from the legal document to form the legal document summary."""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in legal document analysis."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0  # Set to 0 for consistent, factual extraction
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {e}"

# Example Usage:
document = """[Insert your long legal text here]"""
summary = generate_legal_summary(document)
print(summary)
