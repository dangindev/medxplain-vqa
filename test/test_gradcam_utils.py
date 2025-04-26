# test/test_gradcam_utils.py
import torch, torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from src.vqa.gradcam_utils import get_cam_and_boxes

def test_cam():
    model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    img = Image.open("data/images/test/test_0001.jpg").convert("RGB")
    tensor = T.Compose([T.Resize((224,224)),T.ToTensor()])(img).unsqueeze(0)
    cam224, cam_full, boxes = get_cam_and_boxes(model,[model.layer4[-1]],img,tensor)
    assert cam224.shape==(224,224,3)
    assert cam_full.shape==img.size[::-1]
    assert len(boxes)>=1
if __name__=="__main__":
    test_cam()
    print("✅ gradcam_utils OK")
