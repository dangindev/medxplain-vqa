# MedXplain-VQA Project

Dự án MedXplain-VQA là một hệ thống Visual Question Answering (VQA) kết hợp với khả năng giải thích kết quả thông qua Grad-CAM và Chain-of-Thought reasoning.

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd medxplain-vqa
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

```
medxplain-vqa/
├── configs/               # Thư mục chứa file cấu hình
├── data/                  # Dữ liệu đầu vào và kết quả
├── checkpoints/           # Lưu trữ các mô hình đã huấn luyện
├── logs/                  # File log của hệ thống
├── outputs/              # Kết quả đầu ra từ mô hình
├── results/              # Kết quả phân tích và đánh giá
├── src/                  # Mã nguồn chính
│   ├── models/           # Các mô hình ML
│   ├── explainability/   # Các phương pháp giải thích
│   ├── utils/           # Tiện ích và hàm hỗ trợ
│   └── preprocessing/    # Xử lý dữ liệu đầu vào
└── scripts/             # Scripts chạy hệ thống
```

## Hướng dẫn chạy

### 1. Chế độ cơ bản (Basic mode)
Chế độ này chỉ sử dụng BLIP và Gemini để trả lời câu hỏi:
```bash
python scripts/medxplain_vqa.py --mode basic \
    --config configs/config.yaml \
    --model-path checkpoints/blip/checkpoints/best_hf_model \
    --output-dir data/medxplain_basic_results
```

### 2. Chế độ giải thích (Explainable mode)
Chế độ này thêm khả năng giải thích kết quả thông qua Grad-CAM:
```bash
python scripts/medxplain_vqa.py --mode explainable \
    --config configs/config.yaml \
    --model-path checkpoints/blip/checkpoints/best_hf_model \
    --output-dir data/medxplain_explainable_results
```

### 3. Chế độ nâng cao (Enhanced mode)
Chế độ này bổ sung Chain-of-Thought reasoning:
```bash
python scripts/medxplain_vqa.py --mode enhanced \
    --config configs/config.yaml \
    --model-path checkpoints/blip/checkpoints/best_hf_model \
    --output-dir data/medxplain_enhanced_results
```

### Các tùy chọn bổ sung

#### Test với một hình ảnh và câu hỏi cụ thể:
```bash
python scripts/medxplain_vqa.py --mode enhanced \
    --config configs/config.yaml \
    --model-path checkpoints/blip/checkpoints/best_hf_model \
    --image path/to/your/image.jpg \
    --question "Your question here?"
```

#### Test với số lượng mẫu cụ thể:
```bash
python scripts/medxplain_vqa.py --mode enhanced \
    --config configs/config.yaml \
    --model-path checkpoints/blip/checkpoints/best_hf_model \
    --num-samples 5
```

#### Test với bounding box:
```bash
python scripts/medxplain_vqa.py --mode enhanced \
    --enable-bbox \
    --config configs/config.yaml \
    --model-path checkpoints/blip/checkpoints/best_hf_model
```

#### Test đầy đủ tính năng:
```bash
python scripts/medxplain_vqa.py --mode enhanced \
    --enable-bbox \
    --enable-cot \
    --config configs/config.yaml \
    --model-path checkpoints/blip/checkpoints/best_hf_model \
    --output-dir data/medxplain_full_results
```

## Các tham số dòng lệnh

- `--config`: Đường dẫn đến file cấu hình (mặc định: configs/config.yaml)
- `--model-path`: Đường dẫn đến checkpoint của mô hình BLIP
- `--image`: Đường dẫn đến hình ảnh cụ thể (tùy chọn)
- `--question`: Câu hỏi cụ thể (tùy chọn)
- `--num-samples`: Số lượng mẫu test (mặc định: 1)
- `--output-dir`: Thư mục lưu kết quả
- `--mode`: Chế độ xử lý (basic/explainable/enhanced)
- `--enable-cot`: Bật Chain-of-Thought reasoning
- `--enable-bbox`: Bật trích xuất bounding box

## Kết quả đầu ra

Mỗi lần chạy sẽ tạo ra:
1. Hình ảnh visualization (.png)
2. Metadata chi tiết (.json)
3. Log file trong thư mục logs

## Lưu ý quan trọng

1. Đảm bảo đã cài đặt đầy đủ các dependencies trong `requirements.txt`
2. File cấu hình `configs/config.yaml` phải tồn tại và được cấu hình đúng
3. Đường dẫn model checkpoint phải tồn tại
4. Thư mục output sẽ được tạo tự động nếu chưa tồn tại

## Xử lý lỗi

Nếu gặp lỗi:
1. Kiểm tra log file trong thư mục logs
2. Đảm bảo các đường dẫn trong config file là chính xác
3. Kiểm tra quyền truy cập vào các thư mục
4. Xác nhận GPU có sẵn nếu sử dụng CUDA 