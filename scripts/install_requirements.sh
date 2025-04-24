#!/bin/bash

# 1. Cài torch trước
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Cài đúng transformers để tương thích với LAVIS v1.0.2
pip install transformers==4.26.1

# 3. Cài LAVIS
pip install git+https://github.com/salesforce/LAVIS.git@v1.0.2

# 4. Các thư viện khác
pip install bitsandbytes deepspeed peft==0.10.0
pip install opendatasets scikit-learn pandas tqdm pillow
pip install grad-cam==1.5.5

pip install google-generativeai
