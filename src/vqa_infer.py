from vqa.blip2_model import BLIP2VQA
from vqa.gradcam_utils import apply_gradcam, draw_bounding_box

if __name__ == "__main__":
    image_path = "data/images/test/test_0001.jpg"
    question = "What organ is shown in the image?"
    save_path = "outputs/heatmaps/test_0001_heatmap.jpg"

    vqa_model = BLIP2VQA(model_name="blip2_opt", model_type="pretrain_opt2.7b")

    answer = vqa_model.answer(image_path, question)

    print(f"📷 Image: {image_path}")
    print(f"❓ Question: {question}")
    print(f"✅ Answer: {answer}")


    # Sau khi có kết quả từ mô hình
    visualization = apply_gradcam(model, target_layer, input_tensor, original_image)
    box_coords = (50, 50, 200, 200)  # Ví dụ, cần thay bằng tọa độ thực tế
    final_image = draw_bounding_box(visualization, box_coords)
    final_image.save('outputs/visualizations/result.jpg')
