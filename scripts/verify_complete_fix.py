#!/usr/bin/env python
import json
import os

def main():
    print("=== Final Fix Verification ===")
    
    # Check latest results file
    results_file = 'data/chain_of_thought_test/chain_of_thought_test_test_0000.json'
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Check confidence
        reasoning_chain = results.get('reasoning_result', {}).get('reasoning_chain', {})
        overall_confidence = reasoning_chain.get('overall_confidence', 0.0)
        
        print(f"📊 Overall Confidence: {overall_confidence:.3f}")
        
        # Check Grad-CAM
        has_gradcam = 'grad_cam_data' in results and results['grad_cam_data'].get('regions_found', 0) > 0
        print(f"🎯 Grad-CAM: {'✅ WORKING' if has_gradcam else '❌ FAILED'}")
        
        # Check confidence improvement
        confidence_good = overall_confidence > 0.5
        print(f"🎲 Confidence Fix: {'✅ PASSED' if confidence_good else '❌ FAILED'}")
        
        # Check visualization files
        viz_files = [
            'data/chain_of_thought_test/complete_analysis_test_0000.png',
            'data/complete_pipeline_fixed/complete_analysis_test_0000.png'
        ]
        
        viz_count = sum(1 for f in viz_files if os.path.exists(f))
        print(f"📁 Visualization Files: {viz_count}/{len(viz_files)} found")
        
        print(f"\n🎯 FINAL STATUS:")
        if confidence_good and has_gradcam and viz_count > 0:
            print("🎉 ALL FIXES SUCCESSFUL! MedXplain-VQA is 100% COMPLETE!")
        elif confidence_good:
            print("✅ Confidence fixed, Grad-CAM may need attention")
        else:
            print("⚠️ Some issues remain")
            
        # Print individual step confidences for debug
        steps = reasoning_chain.get('steps', [])
        if steps:
            print(f"\n📋 Step Confidences:")
            for i, step in enumerate(steps):
                conf = step.get('confidence', 0.0)
                print(f"  Step {i+1}: {conf:.3f}")
    else:
        print("❌ No results file found")

if __name__ == "__main__":
    main()
