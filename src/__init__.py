"""
CropCare AI — Crop Disease Detection and Fertilizer Recommendation System.

A classical machine learning pipeline for plant disease classification
from leaf images and fertilizer recommendation based on soil/climate data.
"""

import os

# Ensure loky does not throw core detection warnings on Windows
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

__version__ = "1.0.0"
__author__ = "[YOUR NAME]"
