"""
CropCare AI — Feature extraction from leaf images.

Extracts a ~105-dimensional feature vector combining color (HSV),
texture (GLCM + LBP), and shape (Hu moments + severity) features.

Implemented in Phase 2.
"""

from __future__ import annotations
