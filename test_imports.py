#!/usr/bin/env python3
"""
Simple test script to verify imports work correctly after Bayesian cleanup.
"""

import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

def test_imports():
    """Test all critical imports."""
    print("Testing imports after Bayesian cleanup...")
    
    try:
        # Test core imports
        print("Testing core imports...")
        from core.physics_features import VoiceRadarPhysics, VoiceRadarInspiredDynamics
        print("✓ VoiceRadarPhysics imported successfully")
        
        from core.audio_utils import load_audio
        print("✓ audio_utils imported successfully")
        
        from core.feature_extractor import ComprehensiveFeatureExtractor
        print("✓ ComprehensiveFeatureExtractor imported successfully")
        
        from core.processing_pipeline import create_standard_pipeline, create_lightweight_pipeline
        print("✓ processing_pipeline imported successfully")
        
        from core.model_loader import DEVICE
        print("✓ model_loader imported successfully")
        
        # Test utils imports
        print("Testing utils imports...")
        from utils.config_loader import settings
        print("✓ config_loader imported successfully")
        
        # Test optional imports (these might not exist, so we catch exceptions)
        try:
            from utils.logging_system import create_project_logger
            print("✓ logging_system imported successfully")
        except ImportError:
            print("⚠ logging_system not available (optional)")
            
        try:
            from utils.security_validator import SecureAudioLoader
            print("✓ security_validator imported successfully")
        except ImportError:
            print("⚠ security_validator not available (optional)")
            
        try:
            from core.batch_processor import BatchProcessor
            print("✓ batch_processor imported successfully")
        except ImportError:
            print("⚠ batch_processor not available (optional)")
        
        print("\n" + "="*50)
        print("✅ All critical imports successful!")
        print("The project is ready to run.")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("There are still import issues that need to be fixed.")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic physics functionality."""
    print("\nTesting basic physics functionality...")
    
    try:
        import torch
        from core.physics_features import VoiceRadarInspiredDynamics
        
        # Create analyzer
        analyzer = VoiceRadarInspiredDynamics(
            embedding_dim=1024,
            audio_sr=16000
        )
        print("✓ VoiceRadarInspiredDynamics created successfully")
        
        # Create test embeddings
        test_embeddings = torch.randn(50, 1024)
        print("✓ Test embeddings created")
        
        # Test synchronous calculation
        import asyncio
        results = asyncio.run(analyzer.calculate_all_dynamics(test_embeddings))
        print("✓ Physics calculations completed successfully")
        
        print("Results:")
        for key, value in results.items():
            print(f"  {key}: {value.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Physics test failed: {e}")
        return False

if __name__ == '__main__':
    print("="*60)
    print("PHYSICS FEATURES PROJECT - IMPORT TEST")
    print("="*60)
    
    imports_ok = test_imports()
    
    if imports_ok:
        physics_ok = test_basic_functionality()
        
        if physics_ok:
            print("\n🎉 All tests passed! The project is working correctly.")
            sys.exit(0)
        else:
            print("\n⚠ Imports work but physics functionality has issues.")
            sys.exit(1)
    else:
        print("\n❌ Import tests failed. Please fix import issues.")
        sys.exit(1) 