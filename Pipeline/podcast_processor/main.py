# podcast_processor/main.py

import json
import os
import time
import logging
import warnings
import webbrowser
import multiprocessing
from typing import Dict, List

from podcast_processor.config import (
    PLOTS_DIR,
    DEVICE,
    SUPPORTED_EXTENSIONS,
    WHISPER_MODELS,
    AUDIO_DIR,
    TRANSCRIPTIONS_DIR,
    GROUND_TRUTH_NO_ADS_DIR,
    GROUND_TRUTH_ADS_DIR,
    TRANSCRIPT_DIR,
    LLM_MODELS,
    AD_DETECTIONS_DIR
)
from podcast_processor.transcription.transcribe import transcribe_all_models
from podcast_processor.ad_detection.detect_ads import detect_ads_in_transcriptions
from podcast_processor.evaluation.evaluate import (
    aggregate_results,
    save_transcription_results,
    evaluate_ad_detection,
    evaluate_processed_transcriptions
)
from podcast_processor.plotting.plots import (
    plot_transcription_metrics,
    plot_ad_detection_metrics,
    plot_combined_metrics,
    plot_interactive_scatter,
    plot_heatmap,
    plot_ad_detection_precision_recall,
    plot_processed_transcription_wer,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_ad_positions,
    plot_price_vs_ad_accuracy,
    plot_price_vs_time_usage
)
from podcast_processor.reporting.report import (
    generate_summary_report,
    generate_full_transcript_html,
    generate_diff_html,
    load_transcript_segments
)
from podcast_processor.evaluation.utils import (
    load_ground_truth,
    load_ground_truth_ads
)
from podcast_processor.transcription.utils import (
    remove_ads_from_transcription
)

# ---------------------------
# Configuration and Setup
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch.load` with `weights_only=False`"
)

def get_next_run_dir(base_dir: str, prefix: str = "run") -> str:
    """
    Determine the next available run directory (e.g., run_1, run_2, etc.).
    """
    i = 1
    while True:
        run_dir = os.path.join(base_dir, f"{prefix}_{i}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            logger.info(f"Created directory: {run_dir}")
            return run_dir
        i += 1

def main():
    """
    Main function to execute the entire podcast processing pipeline.
    """
    run_dir = get_next_run_dir(PLOTS_DIR, prefix="run")
    logger.info(f"All plots and visualizations will be saved in: {run_dir}")

    start_time = time.time()
    logger.info("Starting transcription...")

    # Step 1: Transcribe all audio files using all Whisper models
    transcription_results = transcribe_all_models(run_dir=run_dir)

    if not transcription_results:
        logger.error("No transcription results to evaluate. Exiting.")
        return

    # Step 2: Aggregate the transcription results
    aggregated = aggregate_results(transcription_results)
    save_transcription_results(aggregated, run_dir=run_dir)

    # Step 3: Detect advertisements in all transcriptions using LLM models
    logger.info("Starting advertisement detection...")
    ad_detections, processing_times = detect_ads_in_transcriptions(run_dir=run_dir)

    # Debugging: Log the ad detections
    logger.debug(f"Ad Detections: {json.dumps(ad_detections, indent=2)}")

    # Step 4: Evaluate Ad Detection
    ground_truth_ads = load_ground_truth_ads(GROUND_TRUTH_ADS_DIR)
    ad_detection_metrics, y_true_all, y_pred_all = evaluate_ad_detection(
        ad_detections, ground_truth_ads, processing_times
    )

    # Debugging: Log ad_detection_metrics
    logger.debug(f"Ad Detection Metrics: {json.dumps(ad_detection_metrics, indent=2)}")

    # Step 5: Remove detected ads from transcriptions and save processed transcriptions
    logger.info("Removing detected ads from transcriptions and evaluating...")
    processed_transcriptions = {}  # Dict[model_name][audio_file] -> processed transcription

    for whisper_model in WHISPER_MODELS:
        whisper_transcription_dir = os.path.join(TRANSCRIPTIONS_DIR, whisper_model)
        if not os.path.exists(whisper_transcription_dir):
            logger.warning(f"Transcription directory for model '{whisper_model}' does not exist. Skipping.")
            continue

        for llm_model_name in LLM_MODELS:
            model_name = f"{whisper_model}_{llm_model_name}"
            processed_transcriptions[model_name] = {}
            whisper_ads_detections = ad_detections.get(whisper_model, {}).get(llm_model_name, {})

            for audio_file in os.listdir(AUDIO_DIR):
                if not audio_file.lower().endswith(SUPPORTED_EXTENSIONS):
                    continue
                transcription_file = os.path.splitext(audio_file)[0] + ".txt"
                transcription_path = os.path.join(whisper_transcription_dir, transcription_file)
                if not os.path.exists(transcription_path):
                    logger.warning(f"Transcription file '{transcription_path}' not found. Skipping.")
                    continue

                # Load the normalized transcription
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    transcription = f.read().strip()
                if not transcription:
                    logger.warning(f"No normalized transcript for '{audio_file}' in model '{whisper_model}'. Skipping.")
                    continue

                # Get detected ads for this audio file and LLM model
                ads = whisper_ads_detections.get(audio_file, [])
                if not ads:
                    logger.info(f"No ads detected for '{audio_file}' using Whisper model '{whisper_model}' and LLM model '{llm_model_name}'. Skipping ad removal.")
                    processed_transcriptions[model_name][audio_file] = transcription
                    continue

                # Remove ads from transcription
                processed_transcription = remove_ads_from_transcription(transcription, ads)
                processed_transcriptions[model_name][audio_file] = processed_transcription

                # Save the processed transcription
                processed_transcription_dir = os.path.join(TRANSCRIPTIONS_DIR, whisper_model, 'processed', llm_model_name)
                os.makedirs(processed_transcription_dir, exist_ok=True)
                processed_transcription_path = os.path.join(processed_transcription_dir, transcription_file)
                with open(processed_transcription_path, 'w', encoding='utf-8') as f:
                    f.write(processed_transcription)
                logger.info(f"Processed transcription saved for '{audio_file}' using Whisper model '{whisper_model}' and LLM model '{llm_model_name}'.")


    # Step 6: Evaluate Processed Transcriptions
    processed_transcription_metrics = evaluate_processed_transcriptions(
        processed_transcriptions, GROUND_TRUTH_NO_ADS_DIR
    )

    # Debugging: Log processed_transcription_metrics
    logger.debug(f"Processed Transcription Metrics: {json.dumps(processed_transcription_metrics, indent=2)}")

    # Step 7: Generate plots based on the aggregated results and ad detections
    plot_transcription_metrics(aggregated, run_dir)
    plot_ad_detection_metrics(ad_detections, run_dir)
    plot_combined_metrics(aggregated, ad_detection_metrics, run_dir)
    plot_interactive_scatter(aggregated, run_dir)
    plot_heatmap(aggregated, run_dir)
    plot_ad_detection_precision_recall(ad_detection_metrics, run_dir)
    plot_processed_transcription_wer(processed_transcription_metrics, run_dir)
    plot_ad_positions(ad_detections, run_dir)
    plot_price_vs_ad_accuracy(ad_detection_metrics, run_dir)
    plot_price_vs_time_usage(ad_detection_metrics, run_dir)

    # Step 8: Generate Full Transcript and Diff HTML for Each Whisper Model, LLM Model, and Audio File
    logger.info("Generating full transcript and diff HTML files...")

    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    # Load ground_truth_ads once before the loop
    ground_truth_ads = load_ground_truth_ads(GROUND_TRUTH_ADS_DIR)

    for whisper_model in WHISPER_MODELS:
        for llm_model_name in LLM_MODELS:
            processed_transcription_dir = os.path.join(TRANSCRIPTIONS_DIR, whisper_model, 'processed', llm_model_name)
            if not os.path.exists(processed_transcription_dir):
                logger.warning(f"Processed transcription directory for model '{whisper_model}' and LLM model '{llm_model_name}' does not exist. Continuing to generate empty transcripts.")
                os.makedirs(processed_transcription_dir, exist_ok=True)

            whisper_ads_detections = ad_detections.get(whisper_model, {}).get(llm_model_name, {})

            for audio_file in audio_files:
                transcription_file = os.path.splitext(audio_file)[0] + ".txt"
                processed_transcription_path = os.path.join(processed_transcription_dir, transcription_file)

                if not os.path.exists(processed_transcription_path):
                    logger.warning(f"Processed transcription file '{processed_transcription_path}' not found. Creating empty transcript.")
                    processed_transcription = ""
                else:
                    # Load the processed transcription
                    try:
                        with open(processed_transcription_path, 'r', encoding='utf-8') as f:
                            processed_transcription = f.read().strip()
                    except Exception as e:
                        logger.error(f"Failed to read processed transcription file '{processed_transcription_path}': {e}")
                        processed_transcription = ""

                # Load the transcript segments (original transcription with ads)
                transcript_segments = load_transcript_segments(whisper_model, audio_file)
                if not transcript_segments:
                    logger.warning(f"No transcript segments found for '{audio_file}' with model '{whisper_model}'. Creating empty segments.")
                    transcript_segments = []

                # Get detected ads for this audio file and LLM model
                ads = whisper_ads_detections.get(audio_file, [])

                # Load ground truth ads for this audio file
                ground_truth_ads_list = ground_truth_ads.get(audio_file, [])

                # Generate Full Transcript HTML with ads highlighted
                generate_full_transcript_html(
                    transcript_segments=transcript_segments,
                    ground_truth_ads=ground_truth_ads_list,
                    detected_ads=ads,
                    model_name=f"{whisper_model}_{llm_model_name}",
                    audio_file=audio_file,
                    run_dir=run_dir
                )

            # Load ground truth transcription without ads
            ground_truth_file = os.path.splitext(audio_file)[0] + ".txt"
            ground_truth_path = os.path.join(GROUND_TRUTH_NO_ADS_DIR, ground_truth_file)
            reference_transcript = load_ground_truth(ground_truth_path)
            if not reference_transcript:
                logger.warning(f"Ground truth transcription not found for '{audio_file}'. Using empty reference transcript.")
                reference_transcript = ""

            hypothesis_transcript = processed_transcription  # The ad-removed transcription

            # Generate Diff HTML
            generate_diff_html(
                reference=reference_transcript,
                hypothesis=hypothesis_transcript,
                model_name=f"{whisper_model}_{llm_model_name}",
                audio_file=audio_file,
                run_dir=run_dir
            )


    # Step 9: Generate Summary Report
    aggregated_ad_detections = {}
    for whisper_model in ad_detections:
        for llm_model_name in ad_detections[whisper_model]:
            model_name = f"{whisper_model}_{llm_model_name}"
            for audio_file, ads in ad_detections[whisper_model][llm_model_name].items():
                if not ads:
                    continue
                if audio_file not in aggregated_ad_detections:
                    aggregated_ad_detections[audio_file] = []
                for ad in ads:
                    # Check if this ad is already in the list (by comparing start and end times)
                    matched = False
                    for existing_ad in aggregated_ad_detections[audio_file]:
                        if abs(existing_ad['start'] - ad['start']) < 1e-3 and abs(existing_ad['end'] - ad['end']) < 1e-3:
                            existing_ad['models'].append(model_name)
                            matched = True
                            break
                    if not matched:
                        # Add the ad with the model name
                        new_ad = {
                            'text': ad.get('text', ''),
                            'start': ad.get('start', 0.0),
                            'end': ad.get('end', 0.0),
                            'models': [model_name]
                        }
                        aggregated_ad_detections[audio_file].append(new_ad)

    logger.info("Generating summary report...")
    generate_summary_report(
        aggregated,
        aggregated_ad_detections,
        ad_detection_metrics,
        processed_transcription_metrics,
        run_dir
    )

    # Automatically open the summary report in the default web browser
    report_path = os.path.join(run_dir, "summary_report.html")
    if os.path.exists(report_path):
        webbrowser.open('file://' + os.path.realpath(report_path))
        logger.info(f"Opened summary report in the default web browser.")
    else:
        logger.warning(f"Summary report not found at '{report_path}'.")

    end_time = time.time()
    total_elapsed = end_time - start_time
    logger.info(f"Podcast processing completed in {total_elapsed:.2f} seconds.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
