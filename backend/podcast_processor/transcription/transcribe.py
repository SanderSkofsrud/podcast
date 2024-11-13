# podcast_processor/transcription/transcribe.py

import os
import time
import logging
import torch
import json
from typing import Tuple, Dict, List
from multiprocessing import Pool, cpu_count, current_process
from functools import partial

import whisper

from podcast_processor.config import (
    WHISPER_MODELS,
    AUDIO_DIR,
    TRANSCRIPT_DIR,
    SUPPORTED_EXTENSIONS,
    DEVICE,
    TRANSCRIPTIONS_DIR
)
from podcast_processor.transcription.models import load_whisper_model
from podcast_processor.transcription.utils import normalize_text, load_ground_truth
from podcast_processor.evaluation.evaluate import calculate_wer
from podcast_processor.reporting.report import generate_diff_html

logger = logging.getLogger(__name__)

def transcribe_audio(model: whisper.Whisper, audio_path: str, model_name: str) -> Tuple[str, float, List[Dict]]:
    """
    Transcribe the given audio file using the provided Whisper model.
    Returns the transcription text, elapsed time, and segments with timestamps.
    """
    try:
        logger.info(f"Process {current_process().name}: Transcribing '{audio_path}' using model '{model_name}'.")
        start_time = time.time()
        result = model.transcribe(
            audio_path,
            verbose=False,
            fp16=(DEVICE == "cuda"),
            condition_on_previous_text=False,
            temperature=0.0,
            task='transcribe',
            language=None  # Auto-detect language
        )
        end_time = time.time()
        transcription = result["text"].strip()
        elapsed_time = end_time - start_time
        segments = result.get("segments", [])
        logger.info(f"Process {current_process().name}: Transcription completed in {elapsed_time:.2f} seconds.")
        return transcription, elapsed_time, segments
    except Exception as e:
        logger.error(f"Process {current_process().name}: Error transcribing '{audio_path}': {e}")
        return "", 0.0, []

def save_transcription(transcription: str, segments: List[Dict], model_name: str, audio_file: str, run_dir: str):
    """
    Save the normalized transcription text and segments to files under the model's directory.
    """
    try:
        normalized_transcription = normalize_text(transcription)
        model_transcription_dir = os.path.join(TRANSCRIPTIONS_DIR, model_name)
        os.makedirs(model_transcription_dir, exist_ok=True)
        transcription_file = os.path.splitext(audio_file)[0] + ".txt"
        transcription_path = os.path.join(model_transcription_dir, transcription_file)
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(normalized_transcription)
        logger.info(f"Process {current_process().name}: Saved normalized transcription to '{transcription_path}'.")

        # Save the transcription segments
        segments_file = os.path.splitext(audio_file)[0] + "_segments.json"
        segments_path = os.path.join(model_transcription_dir, segments_file)
        with open(segments_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=4)
        logger.info(f"Process {current_process().name}: Saved transcription segments to '{segments_path}'.")
    except Exception as e:
        logger.error(f"Process {current_process().name}: Failed to save transcription for '{audio_file}' with model '{model_name}': {e}")

def transcribe_model(model_name: str, run_dir: str, audio_files: List[str], device: str) -> Tuple[str, Dict[str, List[float]]]:
    """
    Transcribe all audio files using a specific Whisper model.
    Returns the model name and a dictionary of speed and accuracy metrics.
    """
    results = {"speed": [], "accuracy": []}
    model = load_whisper_model(model_name, device)
    if model is None:
        logger.error(f"Process {current_process().name}: Failed to load model '{model_name}'.")
        return model_name, results

    for audio_file in audio_files:
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        transcript_file = os.path.splitext(audio_file)[0] + ".txt"
        transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_file)

        if not os.path.exists(transcript_path):
            logger.warning(f"Missing ground truth for '{audio_file}'. Skipping.")
            continue

        transcription, speed, segments = transcribe_audio(model, audio_path, model_name)
        if transcription == "":
            logger.warning(f"Transcription failed for '{audio_file}'. Skipping WER calculation.")
            continue
        results["speed"].append(speed)

        save_transcription(transcription, segments, model_name, audio_file, run_dir)

        ground_truth = load_ground_truth(transcript_path)
        if ground_truth == "":
            logger.warning(f"Ground truth loading failed for '{audio_file}'. Skipping WER calculation.")
            continue

        error_rate = calculate_wer(ground_truth, transcription)
        accuracy = 1.0 - error_rate
        results["accuracy"].append(accuracy)

        logger.info(f"Model: {model_name} | File: {audio_file} | Speed: {speed:.2f}s | Accuracy: {accuracy:.2f}")

        # Generate and save the HTML diff visualization
        generate_diff_html(
            reference=ground_truth,
            hypothesis=transcription,
            model_name=model_name,
            audio_file=audio_file,
            run_dir=run_dir
        )

    return model_name, results

def transcribe_all_models(run_dir: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Transcribe all audio files using all specified Whisper models in parallel.
    Returns a dictionary with model names as keys and speed and accuracy metrics.
    """
    if not WHISPER_MODELS:
        logger.error("No Whisper models specified for transcription.")
        return {}

    # Adjust pool size based on available resources
    pool_size = torch.cuda.device_count() or cpu_count()
    logger.info(f"Starting transcription with {pool_size} parallel processes on '{DEVICE}'.")

    # List all audio files once
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    # Create a list of arguments for each model
    transcribe_args = [(model_name, run_dir, audio_files, DEVICE) for model_name in WHISPER_MODELS]

    with Pool(processes=pool_size) as pool:
        evaluation_results = pool.starmap(transcribe_model, transcribe_args)

    results = {model: {"speed": [], "accuracy": []} for model in WHISPER_MODELS}
    for model_name, res in evaluation_results:
        results[model_name]["speed"].extend(res["speed"])
        results[model_name]["accuracy"].extend(res["accuracy"])

    return results
