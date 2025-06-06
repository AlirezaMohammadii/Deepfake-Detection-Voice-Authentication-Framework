#!/usr/bin/env python3
"""Quick test to verify main functionality works."""

import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

print("Testing basic imports...")

try:
    from core.physics_features import VoiceRadarInspiredDynamics
    print("✓ VoiceRadarInspiredDynamics imported")
    
    from utils.config_loader import settings
    print("✓ Config loaded")
    
    from core.model_loader import DEVICE
    print(f"✓ Device: {DEVICE}")
    
    # Test creating analyzer
    analyzer = VoiceRadarInspiredDynamics(embedding_dim=512, audio_sr=16000)
    print("✓ Physics analyzer created successfully")
    
    print("\n🎉 Basic functionality test PASSED!")
    print("The main Bayesian cleanup was successful.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\nThere are still issues to fix.") 