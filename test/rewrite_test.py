import os
import sys
from dotenv import load_dotenv

# Thêm src/ vào path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Load .env từ thư mục gốc
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path)

from gemini.rewrite import rewrite_question

original_q = "What is shown in the image?"
image_context = "This is a chest X-ray of a 65-year-old male patient with persistent cough."

rewritten_q = rewrite_question(original_q, image_context)
print("🔁 Rewritten:", rewritten_q)
