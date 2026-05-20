"""
Standalone script to ensemble test predictions for a given model pipeline.

Internally calls ``src.cv_ensemble.save_ensemble``; the same function is
also called automatically at the end of run_feature_only.py.

Edit OUTPUT_BASE to target a different model pipeline.

Usage (from project root):
    python scripts/cv_ensemble.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv_ensemble import save_ensemble

# ── paths (edit before running) ──────────────────────────────────────────────
OUTPUT_BASE = 'outputs/feature_only'
N_FOLDS     = 5
THRESHOLD   = 0.5
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    save_ensemble(OUTPUT_BASE, N_FOLDS, THRESHOLD)
