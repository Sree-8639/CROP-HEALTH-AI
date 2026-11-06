"""
backend/ml/crop: extends __path__ to include ROOT/ml/crop so ml.crop modules are discovered.
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ML_CROP_DIR = os.path.join(ROOT_DIR, "ml", "crop")

if os.path.exists(ML_CROP_DIR) and ML_CROP_DIR not in __path__:
    __path__.append(ML_CROP_DIR)
