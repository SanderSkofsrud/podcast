# main.py

import logging
import multiprocessing
import os
import time
import warnings
import webbrowser  # For opening the summary report automatically

from whisper_evaluation.config import PLOTS_DIR
from whisper_evaluation.evaluation import evaluate_models, aggregate_results
from whisper_evaluation.plotting import (
    plot_scatter,
    plot_bar_charts,
    plot_box_plots,
    plot_combined_performance,
    plot_tradeoff_frontier,
    plot_interactive_scatter,
    plot_heatmap,
    generate_summary_report  # New import
)

# ---------------------------
# Configuration and Setup
# ---------------------------

# Configure logging to display time, log level, and message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Suppress specific FutureWarnings related to torch.load
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch.load` with `weights_only=False`"
)

def get_next_run_dir(base_dir: str, prefix: str = "run") -> str:
    """
    Determine the next available run directory (e.g., run_1, run_2, etc.).

    Args:
        base_dir (str): The base directory where run directories are located.
        prefix (str): The prefix for run directories. Defaults to "run".

    Returns:
        str: The path to the next available run directory.
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
    Main function to execute the model evaluation pipeline.
    """
    # Set the multiprocessing start method to 'spawn' to prevent model state inheritance
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Start method has already been set
        pass

    # Determine the run directory
    run_dir = get_next_run_dir(PLOTS_DIR, prefix="run")

    # Log the run directory
    logger.info(f"All plots and visualizations will be saved in: {run_dir}")

    start_time = time.time()
    logger.info("Starting model evaluation...")

    # Step 1: Evaluate models across all audio files
    results = evaluate_models(run_dir=run_dir)

    if not results:
        logger.error("No results to aggregate. Exiting.")
        return

    # Step 2: Aggregate the results to compute average metrics
    aggregated = aggregate_results(results)

    if not aggregated["model"]:
        logger.error("No aggregated data available. Exiting.")
        return

    # Step 3: Plot the aggregated results
    plot_scatter(aggregated, run_dir)

    # Step 4: Plot additional visualizations
    plot_bar_charts(aggregated, run_dir)
    plot_box_plots(results, run_dir)
    plot_combined_performance(aggregated, run_dir)
    plot_tradeoff_frontier(aggregated, run_dir)
    plot_interactive_scatter(aggregated, run_dir)
    plot_heatmap(results, run_dir)

    # Step 5: Generate Summary Report
    generate_summary_report(aggregated, run_dir)

    # Automatically open the summary report in the default web browser
    report_path = os.path.join(run_dir, "summary_report.html")
    if os.path.exists(report_path):
        webbrowser.open('file://' + os.path.realpath(report_path))
        logger.info(f"Opened summary report in the default web browser.")
    else:
        logger.warning(f"Summary report not found at '{report_path}'.")

    end_time = time.time()
    total_elapsed = end_time - start_time
    logger.info(f"Model evaluation completed in {total_elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
