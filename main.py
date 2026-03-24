#!/usr/bin/env python3
# ============================================================
# main.py
# Main driver that coordinates execution of all fragments
# and runs the complete training pipeline for BERT, BART, and GPT
# ============================================================

import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Load all fragments in order
# ============================================================

FRAGMENTS = [
    "00_setup.py",
    "01_batching.py",
    "02_core_modules.py",
    "03_models_bert_bart.py",
    "04_model_gpt_skeleton.py",
    "05_training_utils_and_demos.py",
]

print("Loading fragments...")
for path in FRAGMENTS:
    print(f"  Loading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
        exec(code, globals())

print("\nAll fragments loaded successfully!")
print("Training pipeline complete.")
