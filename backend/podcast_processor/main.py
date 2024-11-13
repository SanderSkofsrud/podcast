# podcast_processor/main.py

import os
import time
import logging
import warnings
import webbrowser
import multiprocessing

from podcast_processor.config import (
    PLOTS_DIR,
    REPORTS_DIR,
    DEVICE,
    SUPPORTED_EXTENSIONS,
    WHISPER_MODELS,
    AUDIO_DIR,
    TRANSCRIPTIONS_DIR,
    GROUND_TRUTH_NO_ADS_DIR,
    GROUND_TRUTH_ADS_DIR, TRANSCRIPT_DIR
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
    plot_processed_transcription_wer
)
from podcast_processor.reporting.report import (
    generate_summary_report,
    generate_full_transcript_html,
    generate_diff_html
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
    ad_detections = detect_ads_in_transcriptions(run_dir=run_dir)

    # Step 3.5: Remove detected ads from transcriptions and evaluate
    logger.info("Removing detected ads from transcriptions and evaluating...")
    processed_transcriptions = {}  # Dict[model_name][audio_file] -> processed transcription
    for model_name in WHISPER_MODELS:
        processed_transcriptions[model_name] = {}
        model_transcription_dir = os.path.join(TRANSCRIPTIONS_DIR, model_name)
        if not os.path.exists(model_transcription_dir):
            logger.warning(f"Transcription directory for model '{model_name}' does not exist. Skipping.")
            continue

        for audio_file in os.listdir(AUDIO_DIR):
            if not audio_file.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            transcription_file = os.path.splitext(audio_file)[0] + ".txt"
            transcription_path = os.path.join(model_transcription_dir, transcription_file)
            if not os.path.exists(transcription_path):
                logger.warning(f"Transcription file '{transcription_path}' not found. Skipping.")
                continue

            # Load the normalized transcription
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription = f.read().strip()
            if not transcription:
                logger.warning(f"No normalized transcript for '{audio_file}' in model '{model_name}'. Skipping.")
                continue

            # Get detected ads
            ads = ad_detections.get(model_name, {}).get(audio_file, [])

            # Remove ads from transcription
            processed_transcription = remove_ads_from_transcription(transcription, ads)
            processed_transcriptions[model_name][audio_file] = processed_transcription

            # Save the processed transcription
            processed_transcription_dir = os.path.join(TRANSCRIPTIONS_DIR, model_name, 'processed')
            os.makedirs(processed_transcription_dir, exist_ok=True)
            processed_transcription_path = os.path.join(processed_transcription_dir, transcription_file)
            with open(processed_transcription_path, 'w', encoding='utf-8') as f:
                f.write(processed_transcription)

    # Step 4: Evaluate Ad Detection and Processed Transcriptions
    ground_truth_ads = load_ground_truth_ads(GROUND_TRUTH_ADS_DIR)
    ad_detection_metrics = evaluate_ad_detection(ad_detections, ground_truth_ads)

    processed_transcription_metrics = evaluate_processed_transcriptions(
        processed_transcriptions, GROUND_TRUTH_NO_ADS_DIR
    )

    # Step 5: Generate plots based on the aggregated results and ad detections
    plot_transcription_metrics(aggregated, run_dir)
    plot_ad_detection_metrics(ad_detections, run_dir)
    plot_combined_metrics(aggregated, ad_detections, run_dir)
    plot_interactive_scatter(aggregated, run_dir)
    plot_heatmap(aggregated, run_dir)
    plot_ad_detection_precision_recall(ad_detection_metrics, run_dir)
    plot_processed_transcription_wer(processed_transcription_metrics, run_dir)

    # Step 6: Generate Full Transcript and Diff HTML for Each Model and Audio File
    logger.info("Generating full transcript and diff HTML files...")

    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    for model_name in WHISPER_MODELS:
        model_transcription_dir = os.path.join(TRANSCRIPTIONS_DIR, model_name)
        if not os.path.exists(model_transcription_dir):
            logger.warning(f"Transcription directory for model '{model_name}' does not exist. Skipping.")
            continue

        for audio_file in audio_files:
            transcription_file = os.path.splitext(audio_file)[0] + ".txt"
            transcription_path = os.path.join(model_transcription_dir, transcription_file)

            if not os.path.exists(transcription_path):
                logger.warning(f"Transcription file '{transcription_path}' not found. Skipping.")
                continue

            # Load the normalized transcription
            try:
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
            except Exception as e:
                logger.error(f"Failed to read transcription file '{transcription_path}': {e}")
                continue

            if not transcript:
                logger.warning(f"No normalized transcript for '{audio_file}' in model '{model_name}'. Skipping.")
                continue

            # Extract detected ads for this audio file and model
            ads = ad_detections.get(model_name, {}).get(audio_file, [])

            # Extract ad texts
            ads_texts = [ad.get('text', '') for ad in ads if isinstance(ad, dict) and 'text' in ad]

            # Generate Full Transcript HTML
            generate_full_transcript_html(
                transcript=transcript,
                ads=ads_texts,
                model_name=model_name,
                audio_file=audio_file,
                run_dir=run_dir
            )

            # Load ground truth transcription
            ground_truth_path = os.path.join(TRANSCRIPT_DIR, os.path.splitext(audio_file)[0] + ".txt")
            reference_transcript = load_ground_truth(ground_truth_path)
            if not reference_transcript:
                logger.warning(f"Ground truth transcription not found for '{audio_file}'. Skipping diff generation.")
                continue

            hypothesis_transcript = transcript  # The model-generated transcript

            # Generate Diff HTML
            generate_diff_html(
                reference=reference_transcript,
                hypothesis=hypothesis_transcript,
                model_name=model_name,
                audio_file=audio_file,
                run_dir=run_dir
            )

    # Step 7: Generate Summary Report
    logger.info("Generating summary report...")
    generate_summary_report(
        aggregated,
        ad_detections,
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
