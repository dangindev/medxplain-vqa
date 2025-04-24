import os
import google.generativeai as genai

# Cấu hình API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Khởi tạo mô hình với tên chính xác
model = genai.GenerativeModel('models/gemini-2.5-pro-preview-03-25')

def rewrite_question(original_question: str, image_context: str = "") -> str:
    prompt = f"""
You are an expert in Visual Question Answering (VQA). Your task is to rewrite the clinical question so that it contains full information, clear and suitable for the context provided.

- ** Original question **: "{original_question}"
- ** Image context **: "{image_context}"

Please rewrite the question:
- Includes all necessary information.
- Clear and specific, avoid vague.
- In accordance with the image context.

Only a question has been rewritten, without any other explanations or suggestions. Write in English
"""
    response = model.generate_content(prompt)
    return response.text.strip()
