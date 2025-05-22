#!/usr/bin/env python
import os
import sys
import subprocess
import json
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_test_configuration(config_name, **kwargs):
    """Run MedXplain-VQA với configuration cụ thể"""
    print(f"\n{'='*60}")
    print(f"TESTING: {config_name}")
    print(f"{'='*60}")
    
    # Prepare command
    cmd = ['python', 'scripts/medxplain_vqa.py']
    
    # Add arguments
    for key, value in kwargs.items():
        if value is True:
            cmd.append(f'--{key.replace("_", "-")}')
        elif value is False:
            cmd.append(f'--disable-{key.replace("_", "-").replace("disable-", "")}')
        else:
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print("❌ FAILED")
            print("Error:", result.stderr[-500:])
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT (5 minutes)")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return False

def main():
    print("MedXplain-VQA Complete Integration Test")
    print("Testing different configurations...")
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    # Test configurations
    configurations = [
        {
            'name': 'Full Pipeline (CoT + GradCAM)',
            'kwargs': {
                'num_samples': 1,
                'enable_cot': True,
                'enable_gradcam': True,
                'output_dir': 'data/test_full_pipeline'
            }
        },
        {
            'name': 'CoT Only (No GradCAM)',
            'kwargs': {
                'num_samples': 1,
                'enable_cot': True,
                'enable_gradcam': False,
                'output_dir': 'data/test_cot_only'
            }
        },
        {
            'name': 'GradCAM Only (No CoT)',
            'kwargs': {
                'num_samples': 1,
                'enable_cot': False,
                'enable_gradcam': True,
                'output_dir': 'data/test_gradcam_only'
            }
        },
        {
            'name': 'Basic Pipeline (No CoT, No GradCAM)',
            'kwargs': {
                'num_samples': 1,
                'enable_cot': False,
                'enable_gradcam': False,
                'output_dir': 'data/test_basic_only'
            }
        }
    ]
    
    # Run tests
    results = {}
    for config in configurations:
        success = run_test_configuration(config['name'], **config['kwargs'])
        results[config['name']] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for config_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{config_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Integration successful!")
        return 0
    else:
        print("⚠️  Some tests failed. Check logs above.")
        return 1

if __name__ == "__main__":
    exit(main())
