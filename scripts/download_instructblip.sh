python - <<'PY'
from lavis.models import load_model_and_preprocess
load_model_and_preprocess("instructblip","llama2_7b", is_eval=True, device="cpu")
print("✅ Weight cached in ~/.cache/lavis")
PY
