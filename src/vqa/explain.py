# src/vqa/explain.py
import os, json, uuid, numpy as np
from PIL import Image
from .gradcam_utils import generate_gradcam, draw_bounding_box

def generate_explanation(sample: dict,
                         cam_method: str = "campp",
                         thresh: float = 0.5):
    """
    sample = {
        'image_path': str,
        'answer'    : str
    }
    returns mention_dict
    """
    img_path   = sample["image_path"]
    answer     = sample["answer"]
    img        = Image.open(img_path).convert("RGB").resize((224,224))

    # ----- Grad-CAM (+) -----
    from .gradcam_utils import GradCAMPlusPlus, GradCAM   # lazy-import lớp
    method_cls = GradCAMPlusPlus if cam_method=="campp" else GradCAM
    cam_image, grayscale_cam = generate_gradcam(
        model      = sample["model"],          # model đã đặt sẵn trong caller
        target_layers = sample["target_layers"],
        input_tensor  = sample["input_tensor"],
        rgb_img       = np.array(img)/255.0,
        cam_class     = method_cls
    )

    # paths
    stem       = os.path.splitext(os.path.basename(img_path))[0]
    uid        = uuid.uuid4().hex[:6]
    cam_jpg    = f"outputs/gradcam/{stem}_{uid}.jpg"
    cam_npy    = f"outputs/cam_arrays/{stem}_{uid}.npy"
    bbox_jpg   = f"outputs/bounding_boxes/{stem}_{uid}.jpg"

    os.makedirs(os.path.dirname(cam_jpg),  exist_ok=True)
    os.makedirs(os.path.dirname(cam_npy),  exist_ok=True)
    os.makedirs(os.path.dirname(bbox_jpg), exist_ok=True)

    # save cam jpg + npy
    Image.fromarray(cam_image).save(cam_jpg)
    np.save(cam_npy, grayscale_cam)

    # ----- Bounding box -----
    img_with_box, bbox = draw_bounding_box(Image.fromarray(cam_image),
                                           grayscale_cam,
                                           threshold=thresh)
    img_with_box.save(bbox_jpg)

    # ----- mention JSON -----
    mention = {
        "image_path" : img_path,
        "answer"     : answer,
        "bounding_box": bbox if bbox else [],
        "cam_method" : cam_method,
        "cam_path"   : cam_jpg,
        "cam_npy"    : cam_npy,
        "bbox_path"  : bbox_jpg,
        "threshold"  : thresh,
    }

    mjson = f"outputs/mentions/{stem}_{uid}.json"
    os.makedirs(os.path.dirname(mjson), exist_ok=True)
    with open(mjson, "w") as f:
        json.dump(mention, f, indent=4)

    return mention
