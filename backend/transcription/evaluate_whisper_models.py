import os
import time
import logging
import warnings
import torch
import whisper
import editdistance  # New import for WER calculation
import matplotlib.pyplot as plt
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch.load` with `weights_only=False`"
)

# Define Whisper model sizes to evaluate
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

# Paths to data
AUDIO_DIR = "data/audio"               # Directory containing podcast audio files
TRANSCRIPT_DIR = "data/transcripts"    # Directory containing ground truth transcriptions

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

def load_model(model_name, device):
    """
    Load the specified Whisper model.
    """
    logger.info(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name, device=device)
    return model

def transcribe_audio(model, audio_path):
    """
    Transcribe the given audio file using the specified Whisper model.
    Returns the transcription and the time taken.
    """
    try:
        logger.info(f"Transcribing audio file: {audio_path} using model.")
        start_time = time.time()
        result = model.transcribe(
            audio_path,
            verbose=False,
            fp16=(device == "cuda"),
            condition_on_previous_text=False,
            temperature=0.0,
            without_timestamps=False,
            task='transcribe'
        )
        end_time = time.time()
        transcription = result["text"].strip()
        elapsed_time = end_time - start_time
        logger.info(f"Transcription completed in {elapsed_time:.2f} seconds.")
        return transcription, elapsed_time
    except Exception as e:
        logger.error(f"Error during transcription of file {audio_path}: {e}")
        return "", 0

def load_ground_truth(transcript_path):
    """
    Load the ground truth transcription from a text file.
    """
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error loading ground truth from {transcript_path}: {e}")
        return ""

def normalize_text(text):
    """
    Normalize text by converting to lowercase and removing punctuation.
    """
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) using editdistance with normalization.
    """
    r = normalize_text(reference).split()
    h = normalize_text(hypothesis).split()
    distance = editdistance.eval(r, h)
    wer_score = distance / len(r) if len(r) > 0 else 0
    return wer_score

def evaluate_models():
    """
    Evaluate different Whisper models on the dataset.
    """
    results = {model: {"speed": [], "accuracy": []} for model in WHISPER_MODELS}

    # Iterate over each model
    for model_name in WHISPER_MODELS:
        model = load_model(model_name, device)

        # Iterate over each audio file
        for audio_file in os.listdir(AUDIO_DIR):
            if not audio_file.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
                continue  # Skip non-audio files

            audio_path = os.path.join(AUDIO_DIR, audio_file)
            transcript_file = os.path.splitext(audio_file)[0] + ".txt"
            transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_file)

            if not os.path.exists(transcript_path):
                logger.warning(f"Missing ground truth for {audio_file}. Skipping.")
                continue

            # Transcribe the audio
            transcription, speed = transcribe_audio(model, audio_path)
            if transcription == "":
                logger.warning(f"Transcription failed for {audio_file}. Skipping WER calculation.")
                continue
            results[model_name]["speed"].append(speed)

            # Load ground truth
            ground_truth = load_ground_truth(transcript_path)
            if ground_truth == "":
                logger.warning(f"Ground truth loading failed for {audio_file}. Skipping WER calculation.")
                continue

            # Calculate Word Error Rate
            error_rate = calculate_wer(ground_truth, transcription)
            accuracy = 1 - error_rate  # Higher is better
            results[model_name]["accuracy"].append(accuracy)

            logger.info(f"Model: {model_name} | File: {audio_file} | Speed: {speed:.2f}s | Accuracy: {accuracy:.2f}")

    return results

def aggregate_results(results):
    """
    Aggregate the results by computing the average speed and accuracy for each model.
    """
    aggregated = {"model": [], "avg_speed": [], "avg_accuracy": []}
    for model, metrics in results.items():
        if metrics["speed"] and metrics["accuracy"]:
            avg_speed = sum(metrics["speed"]) / len(metrics["speed"])
            avg_accuracy = sum(metrics["accuracy"]) / len(metrics["accuracy"])
            aggregated["model"].append(model)
            aggregated["avg_speed"].append(avg_speed)
            aggregated["avg_accuracy"].append(avg_accuracy)
            logger.info(f"Model: {model} | Avg Speed: {avg_speed:.2f}s | Avg Accuracy: {avg_accuracy:.2f}")
        else:
            logger.warning(f"No complete data for model: {model}.")
    return aggregated

def plot_results(aggregated):
    """
    Plot speed vs. accuracy for each model.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(aggregated["avg_speed"], aggregated["avg_accuracy"], color='blue')

    for i, model in enumerate(aggregated["model"]):
        plt.text(aggregated["avg_speed"][i], aggregated["avg_accuracy"][i], model)

    plt.xlabel("Average Transcription Time (s)")
    plt.ylabel("Average Accuracy (1 - WER)")
    plt.title("Whisper Models: Speed vs. Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("whisper_models_speed_vs_accuracy.png")
    plt.show()

if __name__ == "__main__":
    results = evaluate_models()
    aggregated = aggregate_results(results)
    plot_results(aggregated)
