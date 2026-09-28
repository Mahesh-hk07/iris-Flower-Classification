"""
==============================================================================
Multi-Class Flower Image Classifier - Evaluation Entrypoint
==============================================================================
This script evaluates the trained PyTorch MobileNetV2 CNN model checkpoint
(models/flower_classifier.pt) against the held-out test dataset (data/test).

Compatible with both 'python evaluate.py' and 'python evaluate_model.py'.
==============================================================================
"""

import sys
from evaluate import main

if __name__ == "__main__":
    main()
