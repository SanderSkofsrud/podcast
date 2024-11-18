import matplotlib.pyplot as plt
import os
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def plot_metrics(results: Dict[str, Dict[str, float]], run_dir: str):
    """
    Plots precision, recall, F1-score, and average inference time for each model.

    Args:
        results (dict): Dict containing model names as keys and metrics as values.
        run_dir (str): The directory to save the plot.
    """
    models = list(results.keys())
    precision = [results[model]['precision'] for model in models]
    recall = [results[model]['recall'] for model in models]
    f1_scores = [results[model]['f1_score'] for model in models]
    avg_time = [results[model]['average_time'] for model in models]

    x = range(len(models))

    # Plot metrics (Precision, Recall, F1-score)
    plt.figure(figsize=(15, 7))  # Make the figure wider
    plt.bar(x, precision, width=0.2, label='Precision')
    plt.bar([i + 0.2 for i in x], recall, width=0.2, label='Recall')
    plt.bar([i + 0.4 for i in x], f1_scores, width=0.2, label='F1-score')
    plt.ylabel('Score')
    plt.xlabel('Models')
    plt.title('Evaluation of Llama and GPT Models')
    plt.xticks([i + 0.2 for i in x], models, rotation=45, ha="right")  # Rotate labels
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "llama_gpt_models_evaluation.png")
    plt.savefig(plot_path)
    plt.close()  # Close the first figure

    logger.info(f"Plot saved as '{plot_path}'.")

    # Plot inference time as a separate figure
    plt.figure(figsize=(15, 7))
    plt.bar(models, avg_time, color='purple')
    plt.ylabel('Average Inference Time (s)')
    plt.xlabel('Models')
    plt.title('Average Inference Time for Llama and GPT Models')
    plt.xticks(rotation=45, ha="right")  # Rotate labels for time plot as well
    plt.tight_layout()
    time_plot_path = os.path.join(run_dir, "llama_gpt_models_time.png")
    plt.savefig(time_plot_path)
    plt.close()  # Close the second figure

    logger.info(f"Plot saved as '{time_plot_path}'.")
