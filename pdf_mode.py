import fitz  # PyMuPDF

def extract_subtopics_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    subtopics = []

    for page in doc:
        text = page.get_text("text")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        # Check if there are at least 2 lines
        if len(lines) >= 2:
            subtopic = lines[1]  # Second line is the topic name
            subtopics.append(subtopic)

    return list(set(subtopics))  # remove duplicates


def generate_roadmap_from_pdf(pdf_path, generate_fn):
    subtopics = extract_subtopics_from_pdf(pdf_path)
    
    if not subtopics:
        print("❌ No subtopics found in the PDF.")
        return []

    unique_subtopics = list(set(subtopics))
    
    prompt = f"""You are an AI assistant tasked with generating structured learning roadmap steps.

For each subtopic below, generate a roadmap step in **exactly** the following format:

<Subtopic Name>  
<One-line description of what the learner will gain from learning this>  
Time: <estimated days to learn this topic>  
Link: <a relevant W3Schools or GeeksforGeeks link>  
YouTube: <a relevant YouTube video link for this topic>

Here are the subtopics (do not skip any, and remove duplicates automatically):

{chr(10).join(f"- {topic}" for topic in unique_subtopics)}

⚠️ Important:
- Stick to the exact format and avoid extra commentary.
- Each subtopic should be followed by a structured block exactly as shown.
- Prefer GeeksforGeeks or W3Schools links for the 'Link' section.
- Ensure YouTube links are relevant to the subtopic and educational.
"""
    output = generate_fn(prompt)
    return output
