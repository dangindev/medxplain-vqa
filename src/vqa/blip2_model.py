from PIL import Image
import torch
from lavis.models import load_model_and_preprocess
# from .gradcam_utils import generate_heatmap_and_bbox

class BLIP2VQA:
    def __init__(self, model_name: str, model_type: str):
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name=model_name, model_type=model_type, is_eval=True, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def answer(self, image_path, question: str):
        # Load & preprocess image
        raw_image = Image.open(image_path).convert("RGB")
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)

        # Preprocess question
        question_proc = self.txt_processors["eval"](question)

        # Wrap sample
        sample = {
            "image": image,
            "text_input": question_proc
        }

        # Generate answer (Lavis trả về string luôn)
        answers = self.model.generate(sample)
        return answers[0]


