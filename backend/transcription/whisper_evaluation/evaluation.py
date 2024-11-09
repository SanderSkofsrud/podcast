# whisper_evaluation/evaluation.py

import os
import time
import logging
import torch
import whisper
import editdistance
from multiprocessing import Pool, cpu_count, current_process
from typing import Tuple, Dict, List
from whisper_evaluation.config import (
    WHISPER_MODELS,
    AUDIO_DIR,
    TRANSCRIPT_DIR,
    SUPPORTED_EXTENSIONS,
    DEVICE,
    TRANSCRIPTIONS_DIR
)
from whisper_evaluation.utils import load_ground_truth, normalize_text
from whisper_evaluation.models import load_model
from whisper_evaluation.plotting import generate_diff_html

logger = logging.getLogger(__name__)

def transcribe_audio(model: whisper.Whisper, audio_path: str, model_name: str) -> Tuple[str, float]:
    """
    Transcribe the given audio file using the specified Whisper model.

    Args:
        model (whisper.Whisper): The loaded Whisper model for transcription.
        audio_path (str): Path to the audio file to be transcribed.
        model_name (str): Name of the Whisper model being used.

    Returns:
        tuple:
            transcription (str): The transcribed text from the audio.
            elapsed_time (float): Time taken to transcribe the audio in seconds.
    """
    try:
        logger.info(f"Process {current_process().name}: Transcribing '{audio_path}' using model '{model_name}'.")
        start_time = time.time()
        result = model.transcribe(
            audio_path,
            verbose=False,                    # Disable verbose logging from Whisper
            fp16=(DEVICE == "cuda"),          # Use 16-bit floats if using GPU
            condition_on_previous_text=False, # Do not condition on previous text
            temperature=0.0,                  # Deterministic output
            without_timestamps=False,         # Include timestamps in the output
            task='transcribe'                 # Specify task as transcription
            # No explicit language setting to allow auto-detection
        )
        end_time = time.time()
        transcription = result["text"].strip()
        elapsed_time = end_time - start_time
        logger.info(f"Process {current_process().name}: Transcription completed in {elapsed_time:.2f} seconds.")
        return transcription, elapsed_time
    except Exception as e:
        logger.error(f"Process {current_process().name}: Error transcribing '{audio_path}': {e}")
        return "", 0.0

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis transcriptions.

    Args:
        reference (str): The ground truth transcription.
        hypothesis (str): The transcribed text to evaluate.

    Returns:
        float: The calculated WER as a float between 0 and 1.
    """
    # Normalize both reference and hypothesis texts
    r = normalize_text(reference).split()
    h = normalize_text(hypothesis).split()
    # Compute edit distance between the two word lists
    distance = editdistance.eval(r, h)
    # Calculate WER
    wer_score = distance / len(r) if len(r) > 0 else 0.0
    return wer_score

def save_transcription(transcription: str, model_name: str, audio_file: str):
    """
    Save the transcription text to a file under the model's directory.

    Args:
        transcription (str): The transcribed text.
        model_name (str): The name of the Whisper model.
        audio_file (str): The name of the audio file being transcribed.
    """
    try:
        # Ensure the model-specific transcription directory exists
        model_transcription_dir = os.path.join(TRANSCRIPTIONS_DIR, model_name)
        os.makedirs(model_transcription_dir, exist_ok=True)

        # Define the transcription file path
        transcription_file = os.path.splitext(audio_file)[0] + ".txt"
        transcription_path = os.path.join(model_transcription_dir, transcription_file)

        # Save the transcription
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(transcription)

        logger.info(f"Process {current_process().name}: Saved transcription to '{transcription_path}'.")
    except Exception as e:
        logger.error(f"Process {current_process().name}: Failed to save transcription for '{audio_file}' with model '{model_name}': {e}")

def evaluate_model(model_name: str, run_dir: str) -> Tuple[str, Dict[str, List[float]]]:
    """
    Evaluate a single Whisper model across all audio files and generate diff visualizations.

    Args:
        model_name (str): Name of the Whisper model to evaluate.
        run_dir (str): Directory where the visualizations will be saved.

    Returns:
        tuple:
            model_name (str): Name of the Whisper model evaluated.
            results (dict): Dictionary containing lists of 'speed' and 'accuracy'.
    """
    results = {"speed": [], "accuracy": []}

    # Load the model within the child process
    model = load_model(model_name, DEVICE)
    if model is None:
        logger.error(f"Process {current_process().name}: Model '{model_name}' could not be loaded. Skipping evaluation.")
        return model_name, results

    # List all audio files in the AUDIO_DIR with supported extensions
    try:
        audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    except FileNotFoundError:
        logger.error(f"Process {current_process().name}: Audio directory '{AUDIO_DIR}' not found.")
        return model_name, results

    if not audio_files:
        logger.error(f"Process {current_process().name}: No audio files found in the specified directory.")
        return model_name, results

    for audio_file in audio_files:
        # Skip unsupported file types
        if not audio_file.lower().endswith(SUPPORTED_EXTENSIONS):
            logger.debug(f"Process {current_process().name}: Skipping unsupported file type: {audio_file}")
            continue

        audio_path = os.path.join(AUDIO_DIR, audio_file)
        transcript_file = os.path.splitext(audio_file)[0] + ".txt"
        transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_file)

        # Check if the corresponding transcript exists
        if not os.path.exists(transcript_path):
            logger.warning(f"Process {current_process().name}: Missing ground truth for '{audio_file}'. Skipping.")
            continue

        # Transcribe the audio file
        transcription, speed = transcribe_audio(model, audio_path, model_name)
        if transcription == "":
            logger.warning(f"Process {current_process().name}: Transcription failed for '{audio_file}'. Skipping WER calculation.")
            continue
        results["speed"].append(speed)

        # Save the transcription
        save_transcription(transcription, model_name, audio_file)

        # Load ground truth transcription
        ground_truth = load_ground_truth(transcript_path)
        if ground_truth == "":
            logger.warning(f"Process {current_process().name}: Ground truth loading failed for '{audio_file}'. Skipping WER calculation.")
            continue

        # Calculate Word Error Rate
        error_rate = calculate_wer(ground_truth, transcription)
        accuracy = 1.0 - error_rate  # Higher accuracy is better
        results["accuracy"].append(accuracy)

        logger.info(f"Process {current_process().name}: Model: {model_name} | File: {audio_file} | Speed: {speed:.2f}s | Accuracy: {accuracy:.2f}")

        # Generate and save the HTML diff visualization
        generate_diff_html(
            reference=ground_truth,
            hypothesis=transcription,
            model_name=model_name,
            audio_file=audio_file,
            run_dir=run_dir
        )

    return model_name, results

def evaluate_models(run_dir: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Evaluate different Whisper models on the dataset using multiprocessing.

    Args:
        run_dir (str): Directory where visualizations will be saved.

    Returns:
        dict: A dictionary with model names as keys and dictionaries containing
              'speed' and 'accuracy' lists as values.
    """
    if not WHISPER_MODELS:
        logger.error("No Whisper models specified for evaluation.")
        return {}

    # Determine the number of parallel processes based on GPU count or CPU count
    gpu_count = torch.cuda.device_count()
    if gpu_count > 0:
        pool_size = gpu_count
        device = "cuda"
    else:
        pool_size = cpu_count()
        device = "cpu"
        logger.warning("No GPUs available. Falling back to CPU.")

    logger.info(f"Starting evaluation with {pool_size} parallel process(es) on {device}.")

    # Initialize multiprocessing pool and evaluate models
    with Pool(processes=pool_size) as pool:
        # Partial function to include run_dir
        from functools import partial
        evaluate_model_partial = partial(evaluate_model, run_dir=run_dir)

        # Map each model_name to the evaluate_model function
        evaluation_results = pool.map(evaluate_model_partial, WHISPER_MODELS)

    # Compile results into a single dictionary
    results = {model: {"speed": [], "accuracy": []} for model in WHISPER_MODELS}
    for model_name, res in evaluation_results:
        results[model_name]["speed"].extend(res["speed"])
        results[model_name]["accuracy"].extend(res["accuracy"])

    return results

def aggregate_results(results: Dict[str, Dict[str, List[float]]]) -> Dict[str, List]:
    """
    Aggregate the results by computing the average speed and accuracy for each model.

    Args:
        results (dict): Dictionary with model names as keys and dictionaries containing
                        'speed' and 'accuracy' lists as values.

    Returns:
        dict: A dictionary containing lists of 'model', 'avg_speed', and 'avg_accuracy'.
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
