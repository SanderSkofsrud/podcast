# podcast_processor/evaluation/evaluate.py

import logging
from typing import Dict, List
from podcast_processor.evaluation.utils import load_ground_truth
import editdistance
import json
import os
from podcast_processor.config import TRANSCRIPTIONS_DIR
from podcast_processor.reporting.report import convert_time_to_seconds
from podcast_processor.transcription.utils import normalize_text
from podcast_processor.evaluation.utils import load_ground_truth_ads
from thefuzz import fuzz

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

def ads_overlap(ad1, ad2, threshold=0.5):
    latest_start = max(ad1['start'], ad2['start'])
    earliest_end = min(ad1['end'], ad2['end'])
    overlap = max(0, earliest_end - latest_start)
    duration = min(ad1['end'] - ad1['start'], ad2['end'] - ad2['start'])
    return (overlap / duration) >= threshold if duration > 0 else False

def evaluate_ad_detection(detections, ground_truth_ads):
    metrics = {}
    for model_name, model_detections in detections.items():
        tp = fp = fn = 0
        for audio_file, detected_ads in model_detections.items():
            gt_ads = ground_truth_ads.get(audio_file, [])
            matched_ads = set()
            for gt_ad in gt_ads:
                match_found = False
                for det_ad in detected_ads:
                    if ads_overlap(gt_ad, det_ad):
                        tp += 1
                        match_found = True
                        matched_ads.add(id(det_ad))
                        break
                if not match_found:
                    fn += 1
            fp += len(detected_ads) - len(matched_ads)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[model_name] = {'precision': precision, 'recall': recall, 'f1_score': f1_score}
        logger.info(f"Ad Detection Metrics for '{model_name}': Precision={precision:.2f}, Recall={recall:.2f}, F1 Score={f1_score:.2f}")
    return metrics



def evaluate_processed_transcriptions(processed_transcriptions: Dict[str, Dict[str, str]], ground_truth_no_ads_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the processed transcriptions (after ads removed) by computing WER.
    """
    processed_transcription_metrics = {}
    for model_name, model_transcriptions in processed_transcriptions.items():
        total_wer = 0.0
        count = 0
        for audio_file, transcription in model_transcriptions.items():
            # Load ground truth without ads
            ground_truth_file = os.path.splitext(audio_file)[0] + ".txt"
            ground_truth_path = os.path.join(ground_truth_no_ads_dir, ground_truth_file)
            if not os.path.exists(ground_truth_path):
                logger.warning(f"Ground truth without ads not found for '{audio_file}'. Skipping.")
                continue

            ground_truth = load_ground_truth(ground_truth_path)
            if not ground_truth:
                logger.warning(f"No ground truth loaded for '{audio_file}'. Skipping.")
                continue

            # Compute WER
            error_rate = calculate_wer(ground_truth, transcription)
            total_wer += error_rate
            count += 1

        avg_wer = total_wer / count if count > 0 else 0.0
        processed_transcription_metrics[model_name] = {'avg_wer': avg_wer}
        logger.info(f"Processed Transcription - Model: {model_name} | Average WER: {avg_wer:.2f}")

    return processed_transcription_metrics
