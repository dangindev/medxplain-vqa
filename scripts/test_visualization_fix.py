#!/usr/bin/env python
"""
Test script để kiểm tra visualization fix
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def test_single_sample():
    """Test với 1 sample để kiểm tra visualization"""
    
    print("🧪 TESTING VISUALIZATION FIX")
    print("="*50)
    
    # Test với sample đã biết hoạt động
    test_image = 'data/images/test/test_5238.jpg'
    test_question = 'what does this image show?'
    output_dir = 'data/test_viz_fix'
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    # Command để test
    cmd = [
        'python', 'scripts/medxplain_vqa.py',
        '--mode', 'enhanced',
        '--enable-bbox',
        '--enable-cot',
        '--image', test_image,
        '--question', test_question,
        '--output-dir', output_dir
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            print("✅ Command executed successfully")
            
            # Check output files
            json_files = list(Path(output_dir).glob("*.json"))
            png_files = list(Path(output_dir).glob("*.png"))
            
            if json_files and png_files:
                # Load JSON result
                with open(json_files[0]) as f:
                    result_data = json.load(f)
                
                success = result_data.get('success', False)
                bbox_enabled = result_data.get('bbox_enabled', False)
                bbox_count = len(result_data.get('bbox_regions', []))
                grad_cam_available = result_data.get('grad_cam_available', False)
                
                print(f"✅ Success: {success}")
                print(f"✅ Bbox enabled: {bbox_enabled}")
                print(f"✅ Bbox regions: {bbox_count}")
                print(f"✅ Grad-CAM available: {grad_cam_available}")
                print(f"✅ Files: {len(json_files)} JSON, {len(png_files)} PNG")
                
                # Check PNG file size (4-panel should be larger)
                png_file = png_files[0]
                file_size = os.path.getsize(png_file)
                print(f"✅ PNG file size: {file_size/1024/1024:.1f} MB")
                
                # Expected: 4-panel layout should have bbox in filename
                filename = png_file.name
                has_bbox_suffix = "_bbox" in filename
                print(f"✅ Has bbox suffix in filename: {has_bbox_suffix}")
                
                # Summary
                if success and bbox_enabled and bbox_count > 0 and grad_cam_available:
                    print("\n🎉 TEST PASSED - All components working!")
                    print(f"📁 Check visualization: {png_file}")
                    return True
                else:
                    print("\n⚠️ TEST PARTIAL - Some components missing")
                    return False
            else:
                print("❌ No output files generated")
                return False
        else:
            print(f"❌ Command failed: {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_single_sample()
    sys.exit(0 if success else 1) 