# ============================================================
# 99_loader_example.py
# Example main driver.
# This file is complete and can be run directly.
# It loads the fragments into one shared global namespace.
# ============================================================

import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

FRAGMENTS = [
    "fragments/00_setup.py",
    "fragments/01_batching.py",
    "fragments/02_core_modules.py",
    "fragments/03_models_bert_bart.py",
    "fragments/04_model_gpt_skeleton.py",
    "fragments/05_training_utils_and_demos.py",
]

for path in FRAGMENTS:
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), globals())
