import os
import pytest

# Ensure automated test suite runs deterministically against local spectral fixtures
os.environ["DEMO_MODE"] = "true"
