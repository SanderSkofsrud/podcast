# podcast_processor/evaluation/evaluate.py

import logging
from typing import Dict, List
from podcast_processor.evaluation.utils import load_ground_truth
import editdistance
import json
import os
from podcast_processor.config import TRANSCRIPTIONS_DIR
from podcast_processor.transcription.utils import normalize_text

logger = logging.getLogger(__name__)

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis transcriptions.
    """
    r = normalize_text(reference).split()
    h = normalize_text(hypothesis).split()
    distance = editdistance.eval(r, h)
    wer_score = distance / len(r) if len(r) > 0 else 0.0
    return wer_score

def aggregate_results(results: Dict[str, Dict[str, List[float]]]) -> Dict[str, List]:
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

def save_transcription_results(aggregated: dict, run_dir: str):
    """
    Save aggregated transcription results to a JSON file.
    """
    try:
        results_path = os.path.join(run_dir, "transcription_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=4)
        logger.info(f"Saved transcription results to '{results_path}'.")
    except Exception as e:
        logger.error(f"Error saving transcription results: {e}")
