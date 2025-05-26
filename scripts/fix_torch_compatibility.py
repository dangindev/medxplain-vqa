#!/usr/bin/env python
"""
🔧 FIX TORCH COMPATIBILITY: Patch medxplain_vqa.py
"""

def fix_medxplain_vqa():
    """Add torch compatibility fix to main script"""
    
    # Read original file
    with open('scripts/medxplain_vqa.py', 'r') as f:
        content = f.read()
    
    # Find the imports section and add torch fix
    torch_fix = '''
# Fix torch compatibility issue
import torch
if not hasattr(torch, 'get_default_device'):
    torch.get_default_device = lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
    
    # Insert after the first import torch line
    if 'import torch' in content and 'get_default_device' not in content:
        # Find the position after 'import torch'
        import_pos = content.find('import torch')
        next_line_pos = content.find('\n', import_pos) + 1
        
        # Insert the fix
        new_content = content[:next_line_pos] + torch_fix + content[next_line_pos:]
        
        # Write back
        with open('scripts/medxplain_vqa.py', 'w') as f:
            f.write(new_content)
        
        print("✅ Torch compatibility fix applied to medxplain_vqa.py")
    else:
        print("⚠️ File already contains torch fix or no torch import found")

if __name__ == "__main__":
    fix_medxplain_vqa()
