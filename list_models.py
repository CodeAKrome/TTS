#!/usr/bin/env python3
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
n=0
# List available 🐸TTS models and choose the first one
for model_name in TTS().list_models():
    print(f"{n}\t{model_name}")
    n += 1
