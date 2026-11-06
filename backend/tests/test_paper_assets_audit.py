"""
backend/tests/test_paper_assets_audit.py:
Unit test validating that every number in paper_assets/tables matches results/*.json.
"""

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from experiments.check_paper_tables import check_all_paper_tables


def test_paper_assets_tables_match_results_json():
    """Validates that all 114 numbers in paper_assets/tables match results/*.json."""
    # This function raises ValueError / AssertionError on any numerical discrepancy
    check_all_paper_tables()

