import torch
from lavis.models import load_model_and_preprocess

print("✅ Torch OK:", torch.__version__)

model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_vicuna_instruct",  # hoặc "blip2_opt"
    model_type="vicuna7b",         # hoặc "opt2.7b"
    is_eval=True,
    device="cuda"
)

print("✅ BLIP2 loaded")
