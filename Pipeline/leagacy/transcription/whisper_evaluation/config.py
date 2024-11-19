# whisper_evaluation/config.py

import os
import torch

# Define Whisper model sizes to evaluate
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

# Define paths to data directories
AUDIO_DIR = os.path.join("data", "audio")               # Directory containing podcast audio files
TRANSCRIPT_DIR = os.path.join("data", "transcripts")    # Directory containing ground truth transcriptions
TRANSCRIPTIONS_DIR = os.path.join("data", "transcriptions")  # Directory to save model transcriptions

# Define supported audio file extensions
SUPPORTED_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac')

# Define device for model inference (GPU if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define directory to save plots
PLOTS_DIR = "plots"

# Create necessary directories if they don't exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
