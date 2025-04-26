# src/vqa/instructblip_model.py
from lavis.models import load_model_and_preprocess        # :contentReference[oaicite:1]{index=1}
from PIL import Image
import torch

class InstructBLIPVQA:
    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.vis_proc, self.txt_proc = load_model_and_preprocess(
            name="instructblip", model_type="llama2_7b", is_eval=True, device=device
        )                                                # :contentReference[oaicite:2]{index=2}

    def answer(self, image_path: str, question: str, cam_method="campp"):
        from src.vqa.explain import generate_explanation
        raw = Image.open(image_path).convert("RGB")
        image = self.vis_proc["eval"](raw).unsqueeze(0).to(self.device)
        q = self.txt_proc["eval"](question)

        ans = self.model.generate({"image": image, "text_input": q})[0]

        # ----- tạo CAM & bbox -----
        sample = {
            "image_path": image_path,
            "answer": ans,
            "model": self.model.visual_encoder,           # ViT-G/14
            "target_layers": [self.model.visual_encoder.ln_post],
            "input_tensor": image
        }
        mention = generate_explanation(sample, cam_method=cam_method)
        return ans, mention
