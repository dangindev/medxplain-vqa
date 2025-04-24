import os
from dotenv import load_dotenv
import google.generativeai as genai

# ✅ Load từ file .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_rationale(question, answer, bbox_coords=None):
    coord_str = ""
    if bbox_coords:
        coord_str = f"The model focused on the region between coordinates (x0={bbox_coords['x0']}, y0={bbox_coords['y0']}) and (x1={bbox_coords['x1']}, y1={bbox_coords['y1']})."

    prompt = f"""
You are a medical AI assistant. The system answered the following question based on an image:

Question: {question}
Answer: {answer}
{coord_str}

Explain in 2–3 sentences why this answer is likely correct, considering the image-based attention.
Return only the explanation, no extra info.
"""

    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    return response.text.strip()
