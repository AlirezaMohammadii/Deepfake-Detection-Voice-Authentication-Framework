#!/usr/bin/env python3
"""Simple import test script"""

print("🔧 Testing imports...")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers import failed: {e}")

try:
    import pgmpy
    print(f"✅ pgmpy (Bayesian Networks)")
except ImportError as e:
    print(f"❌ pgmpy import failed: {e}")

try:
    from src.core.feature_extractor import ComprehensiveFeatureExtractor
    print("✅ Feature extractor import successful")
except ImportError as e:
    print(f"❌ Feature extractor import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.bayesian import BayesianDeepfakeEngine
    print("✅ Bayesian engine import successful")
except ImportError as e:
    print(f"❌ Bayesian engine import failed: {e}")

print("🏁 Import test completed") 