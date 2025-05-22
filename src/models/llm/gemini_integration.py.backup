import os
import json
import google.generativeai as genai
import logging
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class GeminiIntegration:
    """Module tích hợp Gemini LLM với BLIP cho MedXplain-VQA"""
    
    def __init__(self, config, api_keys_path='configs/api_keys.yaml'):
        """
        Khởi tạo module Gemini
        
        Args:
            config: Cấu hình chính
            api_keys_path: Đường dẫn đến file chứa API key
        """
        self.config = config
        
        # Tải API key
        try:
            from src.utils.config import load_api_keys
            api_keys = load_api_keys(api_keys_path)
            gemini_api_key = api_keys.get('gemini', {}).get('api_key')
            
            if not gemini_api_key:
                raise ValueError("Gemini API key not found in config")
            
            # Cấu hình Gemini
            genai.configure(api_key=gemini_api_key)
            
            # Tạo model Gemini
            model_name = config['model']['llm']['model_name']
            self.model = genai.GenerativeModel(model_name)
            
            # Tham số generation
            self.generation_config = {
                'temperature': config['model']['llm']['temperature'],
                'top_p': config['model']['llm']['top_p'],
                'top_k': config['model']['llm']['top_k'],
                'max_output_tokens': config['model']['llm']['max_output_tokens'],
            }
            
            logger.info(f"Gemini model '{model_name}' initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing Gemini: {e}")
            raise
    
    def encode_image_base64(self, image):
        """
        Mã hóa hình ảnh thành base64 string
        
        Args:
            image: PIL Image
            
        Returns:
            str: Base64 encoded image
        """
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    
    def encode_heatmap_to_base64(self, heatmap, colormap='jet'):
        """
        Mã hóa heatmap thành base64 string
        
        Args:
            heatmap: Numpy array heatmap
            colormap: Colormap để hiển thị heatmap
            
        Returns:
            str: Base64 encoded heatmap image
        """
        # Tạo figure để hiển thị heatmap
        plt.figure(figsize=(5, 5))
        plt.imshow(heatmap, cmap=colormap)
        plt.axis('off')
        
        # Lưu vào buffer
        buffered = BytesIO()
        plt.savefig(buffered, format='JPEG', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Mã hóa base64
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    
    def generate_unified_prompt(self, question, blip_answer, region_descriptions=None):
        """
        Tạo prompt thống nhất để tạo câu trả lời cuối cùng
        
        Args:
            question: Câu hỏi gốc
            blip_answer: Câu trả lời từ BLIP
            region_descriptions: Mô tả các vùng nổi bật (nếu có)
            
        Returns:
            tuple: (system_prompt, prompt)
        """
        system_prompt = """
        You are a medical expert specialized in analyzing pathology images. You're part of the MedXplain-VQA system 
        that combines computer vision and language models for pathology image analysis.
        
        You'll be provided with:
        1. A medical pathology image
        2. A question about the image
        3. An initial analysis from the computer vision component
        4. Highlighted regions of interest in the image (if available)
        
        Your job is to:
        1. Analyze the image
        2. Consider the initial analysis
        3. Pay special attention to the highlighted regions of interest
        4. Provide a single, comprehensive answer that's medically accurate
        5. Focus on what can actually be seen in the image, without speculating
        6. Keep your answer concise but complete
        
        DO NOT mention "BLIP", "regions of interest", "highlighted areas", or any AI systems in your answer. 
        Just provide a fluid, unified medical response that appears to come from a single expert source.
        """
        
        prompt = f"""
        Question: {question}
        
        Initial analysis: {blip_answer}
        """
        
        if region_descriptions:
            prompt += f"\nRegions of interest: {region_descriptions}\n\n"
        
        prompt += "Please provide a single, comprehensive answer that accurately describes what's visible in the image."
        
        return system_prompt, prompt
    
    def generate_unified_answer(self, image, question, blip_answer, heatmap=None, region_descriptions=None):
        """
        Tạo câu trả lời thống nhất kết hợp BLIP và Gemini
        
        Args:
            image: PIL Image
            question: Câu hỏi
            blip_answer: Câu trả lời từ BLIP
            heatmap: Grad-CAM heatmap (nếu có) - ADDED SUPPORT
            region_descriptions: Mô tả các vùng nổi bật (nếu có)
            
        Returns:
            str: Câu trả lời thống nhất
        """
        try:
            # Tạo prompt
            system_prompt, prompt = self.generate_unified_prompt(
                question, 
                blip_answer, 
                region_descriptions
            )
            
            # Chuẩn bị nội dung
            contents = [
                {
                    "role": "user",
                    "parts": [
                        {"text": system_prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": self.encode_image_base64(image)}},
                    ]
                }
            ]
            
            # Thêm heatmap nếu có - NEW FEATURE
            if heatmap is not None:
                try:
                    heatmap_base64 = self.encode_heatmap_to_base64(heatmap)
                    contents[0]["parts"].append(
                        {"text": "A heatmap highlighting regions of interest:"}
                    )
                    contents[0]["parts"].append(
                        {"inline_data": {"mime_type": "image/jpeg", "data": heatmap_base64}}
                    )
                    logger.info("Added heatmap to Gemini input")
                except Exception as e:
                    logger.warning(f"Could not encode heatmap: {e}")
            
            # Thêm prompt
            contents[0]["parts"].append({"text": prompt})
            
            # Gửi request đến Gemini
            response = self.model.generate_content(
                contents=contents,
                generation_config=self.generation_config
            )
            
            # Trả về câu trả lời
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating unified answer with Gemini: {e}")
            return f"Analysis result: {blip_answer} (Enhanced analysis unavailable)"
