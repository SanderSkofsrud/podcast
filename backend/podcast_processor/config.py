# podcast_processor/config.py

import os
import torch

# ---------------------------
# Model Configurations
# ---------------------------

# Define Whisper model sizes to evaluate
WHISPER_MODELS = ["tiny", "small", "large"]

# Define LLM models for ad detection
# LLM_MODELS = ["gpt-4o-mini", "gpt-4-turbo", "gpt-4o"]
LLM_MODELS = ["gpt-4o-mini", "gpt-4o"]

# ---------------------------
# Directory Configurations
# ---------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")                  # Directory containing podcast audio files
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")       # Directory containing ground truth transcriptions
TRANSCRIPTIONS_DIR = os.path.join(DATA_DIR, "transcriptions") # Directory to save model transcriptions
AD_DETECTIONS_DIR = os.path.join(DATA_DIR, "ad_detections")  # Directory to save ad detection results

# Ground truth directories
GROUND_TRUTH_ADS_DIR = os.path.join(DATA_DIR, "ground_truth_ads")  # Directory containing ground truth ads
GROUND_TRUTH_NO_ADS_DIR = os.path.join(DATA_DIR, "ground_truth_no_ads")  # Ground truth transcripts without ads

# Output directories
RESULTS_DIR = os.path.join(BASE_DIR, "results")              # Directory to save all results
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")              # Directory to save generated plots
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")          # Directory to save generated reports

# Supported audio file extensions
SUPPORTED_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac')

# Device configuration for model inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create necessary directories if they don't exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
os.makedirs(AD_DETECTIONS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(GROUND_TRUTH_ADS_DIR, exist_ok=True)
os.makedirs(GROUND_TRUTH_NO_ADS_DIR, exist_ok=True)
