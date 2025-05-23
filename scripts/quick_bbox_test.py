#!/usr/bin/env python
"""
🆕 Quick Bounding Box Test Script
Fast validation of bounding box integration
"""

import os
import sys
import time
from pathlib import Path

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_quick_test():
    """Run quick bounding box integration test"""
    print("🚀 Starting Quick Bounding Box Integration Test")
    
    # Find a test image
    test_images_dir = Path("data/images/test")
    if not test_images_dir.exists():
        print("❌ Test images directory not found")
        return False
    
    # Find first available image
    test_image = None
    for ext in ['.jpg', '.jpeg', '.png']:
        images = list(test_images_dir.glob(f'*{ext}'))
        if images:
            test_image = images[0]
            break
    
    if not test_image:
        print("❌ No test images found")
        return False
    
    print(f"📸 Using test image: {test_image.name}")
    
    # Test with bounding boxes
    print("\n🎯 Testing Enhanced Mode WITH Bounding Boxes...")
    start_time = time.time()
    
    cmd = f'''python scripts/medxplain_vqa.py \
        --image "{test_image}" \
        --question "What does this image show?" \
        --mode enhanced \
        --enable-bbox \
        --output-dir data/quick_bbox_test'''
    
    result = os.system(cmd)
    exec_time = time.time() - start_time
    
    if result == 0:
        print(f"✅ Bounding box test PASSED in {exec_time:.1f}s")
        
        # Check output files
        output_dir = Path("data/quick_bbox_test")
        if output_dir.exists():
            files = list(output_dir.glob("*"))
            print(f"📁 Generated {len(files)} output files:")
            for file in files[:3]:  # Show first 3 files
                print(f"   - {file.name}")
            if len(files) > 3:
                print(f"   ... and {len(files)-3} more files")
        
        return True
    else:
        print(f"❌ Bounding box test FAILED after {exec_time:.1f}s")
        return False

if __name__ == "__main__":
    success = run_quick_test()
    if success:
        print("\n🎉 QUICK TEST PASSED - Bounding box integration working!")
    else:
        print("\n❌ QUICK TEST FAILED - Check logs for details")
    
    sys.exit(0 if success else 1)
