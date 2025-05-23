#!/usr/bin/env python
"""
🆕 COMPLETE INTEGRATION TESTING: Test full MedXplain-VQA pipeline with bounding boxes
"""
import os
import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import setup_logger
import subprocess

def run_integration_test(test_name, command, logger):
    """Run a single integration test"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 RUNNING TEST: {test_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Command: {command}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ TEST PASSED: {test_name} (Duration: {duration:.1f}s)")
            return True, duration, result.stdout
        else:
            logger.error(f"❌ TEST FAILED: {test_name} (Duration: {duration:.1f}s)")
            logger.error(f"Error: {result.stderr}")
            return False, duration, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ TEST TIMEOUT: {test_name} (>300s)")
        return False, 300, "Test timed out"
    except Exception as e:
        logger.error(f"💥 TEST ERROR: {test_name} - {str(e)}")
        return False, 0, str(e)

def main():
    # Setup
    config = Config('configs/config.yaml')
    logger = setup_logger('integration_test', 'logs', level='INFO')
    
    logger.info("🚀 STARTING COMPLETE INTEGRATION TESTING")
    logger.info("Testing all MedXplain-VQA modes with bounding box support")
    
    # Test configurations
    base_command = "python scripts/medxplain_vqa.py --num-samples 1 --output-dir"
    test_configs = [
        {
            'name': 'Basic Mode',
            'command': f"{base_command} data/integration_test/basic --mode basic",
            'expected_files': ['medxplain_basic_*.png', 'medxplain_basic_*.json']
        },
        {
            'name': 'Explainable Mode (No CoT)',
            'command': f"{base_command} data/integration_test/explainable --mode explainable",
            'expected_files': ['medxplain_explainable_*.png', 'medxplain_explainable_*.json']
        },
        {
            'name': 'Enhanced Mode (With CoT)',
            'command': f"{base_command} data/integration_test/enhanced --mode enhanced",
            'expected_files': ['medxplain_enhanced_*.png', 'medxplain_enhanced_*.json']
        },
        {
            'name': 'Explainable + Bounding Boxes',
            'command': f"{base_command} data/integration_test/explainable_bbox --mode explainable --enable-bbox",
            'expected_files': ['medxplain_explainable_bbox_*.png', 'medxplain_explainable_bbox_*.json']
        },
        {
            'name': 'Enhanced + Bounding Boxes (FULL PIPELINE)',
            'command': f"{base_command} data/integration_test/enhanced_bbox --mode enhanced --enable-bbox",
            'expected_files': ['medxplain_enhanced_bbox_*.png', 'medxplain_enhanced_bbox_*.json']
        }
    ]
    
    # Run tests
    results = []
    total_duration = 0
    
    for test_config in test_configs:
        passed, duration, output = run_integration_test(
            test_config['name'], 
            test_config['command'], 
            logger
        )
        
        results.append({
            'name': test_config['name'],
            'passed': passed,
            'duration': duration,
            'output': output
        })
        
        total_duration += duration
        
        # Brief pause between tests
        time.sleep(2)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("🏁 INTEGRATION TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed_tests = sum(1 for r in results if r['passed'])
    total_tests = len(results)
    
    logger.info(f"Tests passed: {passed_tests}/{total_tests}")
    logger.info(f"Total duration: {total_duration:.1f}s")
    logger.info(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    # Detailed results
    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        logger.info(f"{status} {result['name']} ({result['duration']:.1f}s)")
    
    # Check file outputs
    logger.info(f"\n📁 OUTPUT FILE VERIFICATION:")
    for test_config in test_configs:
        output_dir = test_config['command'].split('--output-dir ')[1].split(' ')[0]
        if os.path.exists(output_dir):
            files = list(Path(output_dir).glob('*'))
            logger.info(f"{test_config['name']}: {len(files)} files generated")
        else:
            logger.warning(f"{test_config['name']}: Output directory not found")
    
    # Final verdict
    if passed_tests == total_tests:
        logger.info(f"\n🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("MedXplain-VQA with bounding box support is ready for production!")
        return 0
    else:
        logger.error(f"\n💥 {total_tests - passed_tests} INTEGRATION TESTS FAILED")
        logger.error("Please review failed tests before proceeding.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
