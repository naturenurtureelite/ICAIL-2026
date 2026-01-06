import openai
from openai import OpenAI

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

def three_stage_legal_pipeline(legal_document):
    # --- STAGE 1: RHETORICAL DECOMPOSITION (EXTRACTIVE) ---
    print("Stage 1: Decomposing document into rhetorical segments...")
    stage1_prompt = f"""You are an expert legal summarizer. Your task is to 
    process the following full legal judgment and decompose it into its 
    core rhetorical segments: Facts, Issues Presented, Holding/Ruling, and 
    Court’s Reasoning/Ratio Decidendi. For each segment, you must only 
    extract the most salient, supporting sentences from the original text 
    that capture the essence of that segment. Do not generate any new text.
    
    DOCUMENT:
    {legal_document}"""

    s1_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": stage1_prompt}],
        temperature=0
    )
    extracted_segments = s1_response.choices[0].message.content

    # --- STAGE 2: VALIDATION & CONFIDENCE SCORING ---
    print("Stage 2: Evaluating extraction fidelity...")
    stage2_prompt = f"""Evaluate the set of extracted snippets provided in 
    the previous step. Assign a confidence score (1-5, 5 being highest) for 
    each of the four segments (Facts, Issues, Holding, Reasoning) based on 
    two criteria: Completeness (Does it capture all necessary points?) and 
    Fidelity (Are all snippets direct, unedited extractions from the original text?).
    
    EXTRACTED SNIPPETS:
    {extracted_segments}"""

    s2_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "assistant", "content": extracted_segments},
            {"role": "user", "content": stage2_prompt}
        ],
        temperature=0
    )
    validated_output = s2_response.choices[0].message.content

    # --- STAGE 3: FLUID ABSTRACTIVE SYNTHESIS ---
    print("Stage 3: Generating final abstractive narrative...")
    stage3_prompt = f"""Using only the validated extractive snippets from 
    the final output of the previous step that is given below, generate a 
    single, coherent, and highly-condensed abstractive summary of the 
    legal judgment. The summary must be written in a formal legal style 
    and seamlessly integrate the Facts, Issue, Holding, and Reasoning into 
    a fluid narrative. Focus on creating novel, non-extractive sentences that 
    preserve the factual and legal integrity of the source material.
    
    VALIDATED SNIPPETS:
    {validated_output}"""

    s3_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "assistant", "content": validated_output},
            {"role": "user", "content": stage3_prompt}
        ],
        temperature=0.2 # Lower temperature for formal, non-creative consistency
    )

    return s3_response.choices[0].message.content

# Example Execution
legal_text = "[Full Legal Judgment Text]"
final_narrative = three_stage_legal_pipeline(legal_text)
print("\n--- FINAL LEGAL SUMMARY ---\n")
print(final_narrative)
