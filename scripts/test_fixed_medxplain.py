#!/usr/bin/env python
"""
Test script để kiểm tra medxplain_vqa.py đã được fix
So sánh với test_bounding_box_system.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def test_medxplain_fixed():
    """Test medxplain_vqa.py với fix mới"""
    
    print("🧪 TESTING FIXED MEDXPLAIN-VQA")
    print("="*60)
    
    # Test cases
    test_cases = [
        {
            'image': 'data/images/test/test_5238.jpg',
            'question': 'what does this image show?',
            'name': 'test_5238_working'
        },
        {
            'image': 'data/images/test/test_2253.jpg', 
            'question': 'is female reproductive present?',
            'name': 'test_2253_problematic'
        },
        {
            'image': 'data/images/test/test_0001.jpg',
            'question': 'What does this image show?',
            'name': 'test_0001_standard'
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🔬 TEST {i+1}: {test_case['name']}")
        print("-" * 40)
        
        if not os.path.exists(test_case['image']):
            print(f"❌ Image not found: {test_case['image']}")
            continue
        
        # Run medxplain_vqa with fixed implementation
        cmd = [
            'python', 'scripts/medxplain_vqa.py',
            '--mode', 'enhanced',
            '--enable-bbox',
            '--enable-cot',
            '--image', test_case['image'],
            '--question', test_case['question'],
            '--output-dir', f"data/test_fixed/{test_case['name']}"
        ]
        
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Command executed successfully")
                
                # Check for output files
                output_dir = f"data/test_fixed/{test_case['name']}"
                json_files = list(Path(output_dir).glob("*.json"))
                png_files = list(Path(output_dir).glob("*.png"))
                
                if json_files and png_files:
                    # Load and check JSON result
                    with open(json_files[0]) as f:
                        result_data = json.load(f)
                    
                    success = result_data.get('success', False)
                    bbox_enabled = result_data.get('bbox_enabled', False)
                    bbox_count = len(result_data.get('bbox_regions', []))
                    
                    print(f"✅ Success: {success}")
                    print(f"✅ Bbox enabled: {bbox_enabled}")
                    print(f"✅ Bbox regions: {bbox_count}")
                    print(f"✅ Files generated: {len(json_files)} JSON, {len(png_files)} PNG")
                    
                    results.append({
                        'test_name': test_case['name'],
                        'success': success,
                        'bbox_enabled': bbox_enabled,
                        'bbox_count': bbox_count,
                        'files_generated': len(json_files) + len(png_files)
                    })
                else:
                    print("❌ No output files generated")
                    results.append({
                        'test_name': test_case['name'],
                        'success': False,
                        'error': 'No output files'
                    })
            else:
                print(f"❌ Command failed with return code: {result.returncode}")
                print(f"STDERR: {result.stderr}")
                results.append({
                    'test_name': test_case['name'],
                    'success': False,
                    'error': result.stderr
                })
                
        except subprocess.TimeoutExpired:
            print("❌ Command timed out")
            results.append({
                'test_name': test_case['name'],
                'success': False,
                'error': 'Timeout'
            })
        except Exception as e:
            print(f"❌ Error running command: {e}")
            results.append({
                'test_name': test_case['name'],
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("🎉 TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = sum(1 for r in results if r.get('success', False))
    total_tests = len(results)
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "0%")
    
    print("\nDetailed results:")
    for result in results:
        status = "✅" if result.get('success', False) else "❌"
        print(f"{status} {result['test_name']}: ", end="")
        
        if result.get('success', False):
            bbox_count = result.get('bbox_count', 0)
            print(f"Success, {bbox_count} bounding boxes")
        else:
            error = result.get('error', 'Unknown error')
            print(f"Failed - {error}")
    
    # Save results
    os.makedirs('data/test_fixed', exist_ok=True)
    with open('data/test_fixed/test_summary.json', 'w') as f:
        json.dump({
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests/total_tests if total_tests > 0 else 0,
            'results': results
        }, f, indent=2)
    
    print(f"\n📁 Results saved to: data/test_fixed/test_summary.json")
    
    return successful_tests == total_tests

if __name__ == "__main__":
    success = test_medxplain_fixed()
    sys.exit(0 if success else 1) 