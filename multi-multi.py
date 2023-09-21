#!/usr/bin/env python3

import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models and choose the first one
model_name = TTS().list_models()[1]
# Init TTS

print(f"model\t{model_name}")

tts = TTS(model_name).to(device)

for speaker in tts.speakers:
    print(f"speaker\t{speaker}")

for language in tts.languages:
    print(f"language\t{language}")

# Run TTS
# ❗ Since this model is multi-speaker and multi-lingual, we must set the target speaker and the language
# Text to speech with a numpy output
wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[0], language=tts.languages[0])
# Text to speech to a file
tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")
