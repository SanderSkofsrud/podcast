# podcast_processor/config.py

import os
import torch

# ---------------------------
# Model Configurations
# ---------------------------

# Define Whisper model sizes to evaluate
WHISPER_MODELS = ["tiny", "small", "large"]
# WHISPER_MODELS = ["large"]
# WHISPER_MODELS = ["tiny", "small"]

# Define LLM models for ad detection as a list of model names
# LLM_MODELS = ["gpt-4o-mini"]
# LLM_MODELS = ["gpt-4o"]
# LLM_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
LLM_MODELS = ["gpt-4o-mini", "gpt-4o"]

# Define properties for each LLM model
LLM_MODEL_CONFIGS = {
    "gpt-4o-mini": {
        "max_tokens_per_minute": 200_000,
        "context_window": 128_000,
        "encoding": "cl100k_base",
        "price": 0.15,  # Price per 1M tokens
        "max_output_length": 4096
    },
    "gpt-4o": {
        "max_tokens_per_minute": 30_000,
        "context_window": 32_768,
        "encoding": "cl100k_base",
        "price": 2.50,  # Price per 1M tokens
        "max_output_length": 3072
    },
    "gpt-4-turbo": {
        "max_tokens_per_minute": 30_000,
        "context_window": 32_768,
        "encoding": "cl100k_base",
        "price": 10.00,  # Price per 1M tokens
        "max_output_length": 3072
    }
    # Add more models and their configurations here if needed
}

# ---------------------------
# Directory Configurations
# ---------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")                    # Directory containing podcast audio files
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")         # Directory containing ground truth transcriptions
TRANSCRIPTIONS_DIR = os.path.join(DATA_DIR, "transcriptions")  # Directory to save model transcriptions
AD_DETECTIONS_DIR = os.path.join(DATA_DIR, "ad_detections")    # Directory to save ad detection results

# Ground truth directories
GROUND_TRUTH_ADS_DIR = os.path.join(DATA_DIR, "ground_truth_ads")          # Directory containing ground truth ads
GROUND_TRUTH_NO_ADS_DIR = os.path.join(DATA_DIR, "ground_truth_no_ads")    # Ground truth transcripts without ads

# Output directories
RESULTS_DIR = os.path.join(BASE_DIR, "results")                      # Directory to save all results
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")                       # Directory to save generated plots

# Supported audio file extensions
SUPPORTED_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac')

# Device configuration for model inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create the necessary directories if they don't exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
os.makedirs(AD_DETECTIONS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(GROUND_TRUTH_ADS_DIR, exist_ok=True)
os.makedirs(GROUND_TRUTH_NO_ADS_DIR, exist_ok=True)
