# podcast_processor/main.py

import os
import time
import logging
import warnings
import webbrowser
import multiprocessing  # Ensure multiprocessing is imported

from podcast_processor.config import PLOTS_DIR, REPORTS_DIR, DEVICE
from podcast_processor.transcription.transcribe import transcribe_all_models
from podcast_processor.ad_detection.detect_ads import detect_ads_in_transcriptions
from podcast_processor.evaluation.evaluate import aggregate_results, save_transcription_results
from podcast_processor.plotting.plots import (
    plot_transcription_metrics,
    plot_ad_detection_metrics,
    plot_combined_metrics,
    plot_interactive_scatter,
    plot_heatmap
)
from podcast_processor.reporting.report import (
    generate_summary_report,
    generate_full_transcript_html,
    generate_diff_html  # Ensure generate_diff_html is imported
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

    # Step 4: Generate plots based on the aggregated results and ad detections
    plot_transcription_metrics(aggregated, run_dir)
    plot_ad_detection_metrics(ad_detections, run_dir)
    plot_combined_metrics(aggregated, ad_detections, run_dir)
    plot_interactive_scatter(aggregated, run_dir)
    plot_heatmap(aggregated, run_dir)

    # Step 5: Generate Full Transcript HTML and Diff HTML for Each Model and Audio File
    logger.info("Generating full transcript and diff HTML files...")
    for model_name, audio_files in aggregated.get("model", {}).items():
        for audio_file, transcript_data in audio_files.items():
            # Extract the normalized transcript
            transcript = transcript_data.get("normalized_transcript", "")
            if not transcript:
                logger.warning(f"No normalized transcript for '{audio_file}' in model '{model_name}'. Skipping.")
                continue

            # Extract detected ads for this audio file and model
            ads = ad_detections.get(model_name, {}).get(audio_file, [])

            # Generate Full Transcript HTML
            generate_full_transcript_html(
                transcript=transcript,
                ads=ads,
                model_name=model_name,
                audio_file=audio_file,
                run_dir=run_dir
            )

            # Assuming you have a reference transcription for diffs
            # This could be loaded from a ground truth file or another source
            reference_transcript = load_ground_truth(os.path.join(REPORTS_DIR, audio_file + ".txt"))  # Adjust the path as needed
            hypothesis_transcript = transcript  # The model-generated transcript

            generate_diff_html(
                reference=reference_transcript,
                hypothesis=hypothesis_transcript,
                model_name=model_name,
                audio_file=audio_file,
                run_dir=run_dir
            )

    # Step 6: Generate Summary Report
    logger.info("Generating summary report...")
    generate_summary_report(aggregated, ad_detections, run_dir)

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
    multiprocessing.set_start_method('spawn')  # Ensure proper multiprocessing start method
    main()
