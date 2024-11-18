# podcast_processor/evaluation/evaluate.py

import logging
from typing import Dict, List
from podcast_processor.evaluation.utils import load_ground_truth
import editdistance
import json
import os
from podcast_processor.config import TRANSCRIPTIONS_DIR, LLM_MODELS, WHISPER_MODELS
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

def ads_overlap(ad1, ad2, threshold=0.3):
    """
    Determine if two ads overlap by at least the specified threshold.
    """
    latest_start = max(ad1['start'], ad2['start'])
    earliest_end = min(ad1['end'], ad2['end'])
    overlap = max(0, earliest_end - latest_start)
    duration = min(ad1['end'] - ad1['start'], ad2['end'] - ad2['start'])
    return (overlap / duration) >= threshold if duration > 0 else False

def evaluate_ad_detection(aggregated_ad_detections, ground_truth_ads, processing_times):
    """
    Evaluate ad detection by calculating precision, recall, and F1 score for each combination of Whisper and LLM models.

    Args:
        aggregated_ad_detections (Dict[str, Dict[str, Dict[str, List[Dict]]]]): Detected ads per whisper_model and llm_model.
        ground_truth_ads (Dict[str, List[Dict]]): Ground truth ads per audio file.
        processing_times (Dict[str, float]): Processing time per LLM model.

    Returns:
        metrics (Dict[str, Dict[str, float]]): Evaluation metrics per model combination.
        y_true_all (Dict[str, List[int]]): Ground truth labels per model combination.
        y_pred_all (Dict[str, List[int]]): Prediction labels per model combination.
    """
    metrics = {}
    y_true_all = {}
    y_pred_all = {}

    for whisper_model in WHISPER_MODELS:
        whisper_detections = aggregated_ad_detections.get(whisper_model, {})
        for llm_model_name in LLM_MODELS:
            model_detections = whisper_detections.get(llm_model_name, {})
            tp = 0
            fp = 0
            fn = 0
            y_true = []
            y_pred = []
            logger.info(f"Evaluating model combination '{whisper_model}' and '{llm_model_name}'...")

            for audio_file, detected_ads in model_detections.items():
                gt_ads = ground_truth_ads.get(audio_file, [])
                logger.debug(f"Audio File: {audio_file} | Ground Truth Ads: {len(gt_ads)} | Detected Ads: {len(detected_ads)}")
                logger.debug(f"Ground Truth Ads for '{audio_file}': {gt_ads}")
                logger.debug(f"Detected Ads for '{audio_file}': {detected_ads}")

                # Keep track of which ground truth ads have been matched
                gt_matched = [False] * len(gt_ads)

                # Initialize counts per audio file
                tp_audio = 0
                fp_audio = 0
                fn_audio = 0
                y_true_audio = []
                y_pred_audio = []

                for det_idx, det_ad in enumerate(detected_ads):
                    det_start = det_ad['start']
                    det_end = det_ad['end']
                    det_duration = det_end - det_start

                    best_overlap_ratio = 0.0
                    best_gt_idx = -1

                    for idx, gt_ad in enumerate(gt_ads):
                        if gt_matched[idx]:
                            continue
                        gt_start = gt_ad['start']
                        gt_end = gt_ad['end']
                        gt_duration = gt_end - gt_start

                        overlap = max(0, min(det_end, gt_end) - max(det_start, gt_start))
                        overlap_ratio = overlap / gt_duration if gt_duration > 0 else 0.0

                        if overlap_ratio > best_overlap_ratio:
                            best_overlap_ratio = overlap_ratio
                            best_gt_idx = idx

                    if best_overlap_ratio >= 0.5:
                        gt_ad = gt_ads[best_gt_idx]
                        similarity = fuzz.token_set_ratio(det_ad['text'], gt_ad['text'])
                        if similarity >= 80:
                            # True Positive
                            tp_audio += 1
                            gt_matched[best_gt_idx] = True
                            y_true_audio.append(1)
                            y_pred_audio.append(1)
                            logger.debug(f"TP: Detected Ad {det_idx+1} matches GT Ad {best_gt_idx+1} with overlap ratio {best_overlap_ratio:.2f} and similarity {similarity}.")
                        else:
                            # False Positive due to low text similarity
                            fp_audio += 1
                            y_true_audio.append(0)
                            y_pred_audio.append(1)
                            logger.debug(f"FP: Detected Ad {det_idx+1} overlaps but has low text similarity ({similarity}).")
                    else:
                        # False Positive
                        fp_audio += 1
                        y_true_audio.append(0)
                        y_pred_audio.append(1)
                        logger.debug(f"FP: Detected Ad {det_idx+1} does not match any GT Ad (Best overlap ratio {best_overlap_ratio:.2f})")

                # Any unmatched ground truth ads are False Negatives
                for idx, matched in enumerate(gt_matched):
                    if not matched:
                        fn_audio += 1
                        y_true_audio.append(1)
                        y_pred_audio.append(0)
                        logger.debug(f"FN: GT Ad {idx+1} was not detected.")

                # Update totals
                tp += tp_audio
                fp += fp_audio
                fn += fn_audio
                y_true.extend(y_true_audio)
                y_pred.extend(y_pred_audio)

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            processing_time = processing_times.get(llm_model_name, 0.0)

            metrics_key = f"{whisper_model}_{llm_model_name}"
            metrics[metrics_key] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score_value,
                'processing_time': processing_time
            }

            y_true_all[metrics_key] = y_true
            y_pred_all[metrics_key] = y_pred

            logger.info(f"Ad Detection Metrics for '{metrics_key}': Precision={precision:.2f}, Recall={recall:.2f}, F1 Score={f1_score_value:.2f}, Processing Time={processing_time:.2f}s")

    return metrics, y_true_all, y_pred_all

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

            logger.debug(f"Model: {model_name} | Audio File: {audio_file} | WER: {error_rate:.2f}")

        avg_wer = total_wer / count if count > 0 else 0.0
        processed_transcription_metrics[model_name] = {'avg_wer': avg_wer}
        logger.info(f"Processed Transcription - Model: {model_name} | Average WER: {avg_wer:.2f}")

    return processed_transcription_metrics

