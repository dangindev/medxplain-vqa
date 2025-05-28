#!/usr/bin/env python3
"""
Methodology Information Inspector
Automatically extracts technical details from MedXplain-VQA codebase for paper writing
"""

import os
import sys
import json
import yaml
import re
import inspect
from pathlib import Path
import importlib.util

def print_section(title):
    print(f"\n{'='*60}")
    print(f"📋 {title}")
    print(f"{'='*60}")

def print_subsection(title):
    print(f"\n🔍 {title}")
    print("-" * 40)

def safe_read_file(file_path):
    """Safely read file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading {file_path}: {e}"

def safe_read_yaml(file_path):
    """Safely read YAML file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        return f"Error reading YAML {file_path}: {e}"

def extract_config_info():
    """Extract configuration information"""
    print_section("CONFIGURATION INFORMATION")
    
    # Main config file
    config_file = "configs/config.yaml"
    if os.path.exists(config_file):
        print_subsection("Main Configuration (configs/config.yaml)")
        config = safe_read_yaml(config_file)
        print(json.dumps(config, indent=2, default=str))
    else:
        print(f"⚠️ Config file not found: {config_file}")
    
    # API keys config
    api_config_file = "configs/api_keys.yaml"
    if os.path.exists(api_config_file):
        print_subsection("API Configuration (configs/api_keys.yaml)")
        print("🔒 API keys file exists (content hidden for security)")
    else:
        print(f"⚠️ API config file not found: {api_config_file}")

def extract_model_info():
    """Extract BLIP model configuration"""
    print_section("BLIP-2 MODEL INFORMATION")
    
    # Check model.py for architecture details
    model_file = "src/models/blip2/model.py"
    if os.path.exists(model_file):
        print_subsection("BLIP-2 Model Implementation")
        content = safe_read_file(model_file)
        
        # Extract key patterns
        print("🔍 Searching for key model parameters...")
        
        # Model name/variant
        model_patterns = [
            r'model_name\s*=\s*[\'"]([^\'"]+)[\'"]',
            r'from_pretrained\([\'"]([^\'"]+)[\'"]',
            r'Salesforce/([^\'"\s]+)',
            r'blip[^\'"\s]*'
        ]
        
        for pattern in model_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                print(f"Model references found: {matches}")
        
        # Image size patterns
        size_patterns = [
            r'image_size\s*=\s*(\d+)',
            r'size\s*=\s*\((\d+),\s*(\d+)\)',
            r'resize\s*\(\s*(\d+)'
        ]
        
        for pattern in size_patterns:
            matches = re.findall(pattern, content)
            if matches:
                print(f"Image size references: {matches}")
        
        # Print first 50 lines to see structure
        lines = content.split('\n')
        print(f"\n📄 First 30 lines of {model_file}:")
        for i, line in enumerate(lines[:30]):
            print(f"{i+1:2d}: {line}")
            
    else:
        print(f"⚠️ Model file not found: {model_file}")

def extract_training_info():
    """Extract training configuration"""
    print_section("TRAINING INFORMATION")
    
    # Check trainer.py
    trainer_file = "src/models/blip2/trainer.py"
    if os.path.exists(trainer_file):
        print_subsection("Training Configuration")
        content = safe_read_file(trainer_file)
        
        # Extract training parameters
        training_patterns = {
            'learning_rate': [r'lr\s*=\s*([\d.e-]+)', r'learning_rate\s*=\s*([\d.e-]+)'],
            'batch_size': [r'batch_size\s*=\s*(\d+)', r'per_device_train_batch_size\s*=\s*(\d+)'],
            'epochs': [r'epochs\s*=\s*(\d+)', r'num_train_epochs\s*=\s*(\d+)'],
            'optimizer': [r'optimizer\s*=\s*[\'"]?([^\'"\s,)]+)', r'AdamW|SGD|Adam'],
            'weight_decay': [r'weight_decay\s*=\s*([\d.e-]+)']
        }
        
        for param, patterns in training_patterns.items():
            print(f"\n🔍 Searching for {param}:")
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    print(f"  Found: {matches}")
    
    # Check train_blip.py script
    train_script = "scripts/train_blip.py"
    if os.path.exists(train_script):
        print_subsection("Training Script")
        content = safe_read_file(train_script)
        print("📄 First 50 lines of training script:")
        lines = content.split('\n')
        for i, line in enumerate(lines[:50]):
            if 'lr' in line.lower() or 'batch' in line.lower() or 'epoch' in line.lower():
                print(f"{i+1:2d}: {line}")

def extract_gradcam_info():
    """Extract Grad-CAM implementation details"""
    print_section("GRAD-CAM IMPLEMENTATION")
    
    gradcam_files = [
        "src/explainability/grad_cam.py",
        "src/explainability/enhanced_grad_cam.py",
        "src/explainability/bounding_box_extractor.py"
    ]
    
    for file_path in gradcam_files:
        if os.path.exists(file_path):
            print_subsection(f"File: {file_path}")
            content = safe_read_file(file_path)
            
            # Extract parameter patterns
            param_patterns = {
                'target_layer': [r'layer_name\s*=\s*[\'"]([^\'"]+)[\'"]', r'layers\.(\d+)'],
                'threshold': [r'threshold\s*=\s*([\d.]+)', r'attention_threshold\s*=\s*([\d.]+)'],
                'min_size': [r'min.*size\s*=\s*(\d+)', r'min_region_size\s*=\s*(\d+)'],
                'max_regions': [r'max.*regions?\s*=\s*(\d+)'],
                'expansion': [r'expansion\s*=\s*([\d.]+)', r'box_expansion\s*=\s*([\d.]+)']
            }
            
            for param, patterns in param_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        print(f"  {param}: {matches}")
            
            # Show class definitions
            class_matches = re.findall(r'^class\s+(\w+).*?:', content, re.MULTILINE)
            if class_matches:
                print(f"  Classes defined: {class_matches}")

def extract_cot_info():
    """Extract Chain-of-Thought implementation"""
    print_section("CHAIN-OF-THOUGHT REASONING")
    
    cot_file = "src/explainability/rationale/chain_of_thought.py"
    if os.path.exists(cot_file):
        print_subsection("Chain-of-Thought Implementation")
        content = safe_read_file(cot_file)
        
        # Look for reasoning steps
        step_patterns = [
            r'step.*?[\'"]([^\'"]+)[\'"]',
            r'Step\s+\d+[:\s]*([^\n]+)',
            r'reasoning.*steps.*?=.*?\[(.*?)\]'
        ]
        
        print("🔍 Searching for reasoning steps:")
        for pattern in step_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            if matches:
                print(f"  Steps found: {matches[:10]}")  # Limit output
        
        # Look for confidence calculation
        confidence_patterns = [
            r'confidence.*?=.*?(.*?)[\n;]',
            r'def.*confidence.*?\((.*?)\):',
            r'harmonic.*mean'
        ]
        
        print("\n🔍 Searching for confidence calculation:")
        for pattern in confidence_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                print(f"  Confidence logic: {matches}")

def extract_llm_integration():
    """Extract LLM integration details"""
    print_section("LLM INTEGRATION")
    
    gemini_file = "src/models/llm/gemini_integration.py"
    if os.path.exists(gemini_file):
        print_subsection("Gemini Integration")
        content = safe_read_file(gemini_file)
        
        # Look for model configuration
        model_patterns = [
            r'model[\'"]?\s*[:=]\s*[\'"]([^\'"]+)[\'"]',
            r'gemini[^\'"\s]*',
            r'temperature\s*=\s*([\d.]+)',
            r'max_tokens\s*=\s*(\d+)'
        ]
        
        print("🔍 Searching for LLM configuration:")
        for pattern in model_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                print(f"  Found: {matches}")
        
        # Look for prompt templates
        prompt_patterns = [
            r'prompt\s*=\s*[f]?[\'"]([^\'"]*{[^}]*}[^\'"]*)[\'"]',
            r'template\s*=\s*[\'"]([^\'"]+)[\'"]'
        ]
        
        print("\n🔍 Searching for prompt templates:")
        for pattern in prompt_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                for i, match in enumerate(matches[:3]):  # Show first 3
                    print(f"  Template {i+1}: {match[:200]}...")

def extract_evaluation_info():
    """Extract evaluation metrics implementation"""
    print_section("EVALUATION METRICS")
    
    eval_files = [
        "src/models/blip2/evaluation.py",
        "scripts/paper_evaluation_suite.py"
    ]
    
    for file_path in eval_files:
        if os.path.exists(file_path):
            print_subsection(f"File: {file_path}")
            content = safe_read_file(file_path)
            
            # Look for metric definitions
            metric_patterns = [
                r'def\s+(.*?metric.*?)\(',
                r'def\s+(.*?score.*?)\(',
                r'def\s+(.*?calculate.*?)\(',
                r'medical.*terminology',
                r'clinical.*structure',
                r'coherence'
            ]
            
            print("🔍 Searching for evaluation metrics:")
            for pattern in metric_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    print(f"  Found: {matches}")

def extract_script_parameters():
    """Extract main script parameters"""
    print_section("MAIN SCRIPT PARAMETERS")
    
    main_script = "scripts/medxplain_vqa.py"
    if os.path.exists(main_script):
        print_subsection("Main Script Analysis")
        content = safe_read_file(main_script)
        
        # Look for argument parser
        arg_patterns = [
            r'add_argument\([\'"]([^\'"]+)[\'"].*?default\s*=\s*([^,)]+)',
            r'parser\.add_argument\([^)]+\)'
        ]
        
        print("🔍 Command line arguments:")
        for pattern in arg_patterns:
            matches = re.findall(pattern, content)
            if matches:
                for match in matches[:10]:  # Show first 10
                    print(f"  {match}")
        
        # Look for processing modes
        mode_patterns = [
            r'mode.*?=.*?[\'"]([^\'"]+)[\'"]',
            r'choices\s*=\s*\[(.*?)\]'
        ]
        
        print("\n🔍 Processing modes:")
        for pattern in mode_patterns:
            matches = re.findall(pattern, content)
            if matches:
                print(f"  Modes: {matches}")

def main():
    """Main inspection function"""
    print("🔍 MedXplain-VQA Methodology Information Inspector")
    print("=" * 60)
    print("Analyzing codebase to extract technical details for paper...")
    
    # Check if we're in the right directory
    if not os.path.exists("src") or not os.path.exists("scripts"):
        print("⚠️ Warning: Not in MedXplain-VQA root directory")
        print("Please run this script from the project root directory")
        return
    
    # Extract all information
    extract_config_info()
    extract_model_info()
    extract_training_info()
    extract_gradcam_info()
    extract_cot_info()
    extract_llm_integration()
    extract_evaluation_info()
    extract_script_parameters()
    
    print_section("SUMMARY")
    print("✅ Information extraction completed!")
    print("📝 Use the extracted information above to fill in methodology details")
    print("🔍 For missing information, check the specific files mentioned")

if __name__ == "__main__":
    main()
