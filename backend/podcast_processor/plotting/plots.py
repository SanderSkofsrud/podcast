# podcast_processor/plotting/plots.py

import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List
from podcast_processor.config import WHISPER_MODELS, LLM_MODELS

logger = logging.getLogger(__name__)

def plot_transcription_metrics(aggregated: dict, run_dir: str):
    """
    Plot transcription speed vs. accuracy.
    """
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        aggregated["avg_speed"],
        aggregated["avg_accuracy"],
        color='skyblue',
        s=100,
        edgecolors='k'
    )
    for i, model in enumerate(aggregated["model"]):
        plt.text(
            aggregated["avg_speed"][i] + 0.02,
            aggregated["avg_accuracy"][i] + 0.002,
            model,
            fontsize=10,
            weight='bold',
            verticalalignment='bottom'
        )
    plt.xlabel("Average Transcription Time (s)", fontsize=12)
    plt.ylabel("Average Accuracy (1 - WER)", fontsize=12)
    plt.title("Whisper Models: Speed vs. Accuracy", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "whisper_models_speed_vs_accuracy.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Transcription scatter plot saved as '{plot_path}'.")

def plot_ad_detection_metrics(ad_detections: Dict[str, Dict[str, List[Dict]]], run_dir: str):
    """
    Plot advertisement detection metrics such as number of ads detected per model.
    """
    model_ads_count = {model: 0 for model in ad_detections.keys()}
    for model, detections in ad_detections.items():
        for ads in detections.values():
            model_ads_count[model] += len(ads)

    # Prepare data for plotting
    models = list(model_ads_count.keys())
    ads_counts = [model_ads_count[model] for model in models]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=models, y=ads_counts, palette=sns.color_palette("viridis", n_colors=len(models)))
    plt.xlabel("LLM Models", fontsize=12)
    plt.ylabel("Total Advertisements Detected", fontsize=12)
    plt.title("Advertisements Detected per LLM Model", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "ad_detections_per_model.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Advertisement detection bar plot saved as '{plot_path}'.")

def plot_combined_metrics(aggregated: dict, ad_detections: Dict[str, Dict[str, List[Dict]]], run_dir: str):
    """
    Create combined plots that show transcription and ad detection metrics together.
    """
    # Aggregate ads detected per LLM model
    ads_detected = {model: sum(len(ads) for ads in detections.values()) for model, detections in ad_detections.items()}

    # Create a DataFrame for plotting
    data = []
    for model in aggregated["model"]:
        speed = aggregated["avg_speed"][aggregated["model"].index(model)]
        accuracy = aggregated["avg_accuracy"][aggregated["model"].index(model)]
        ads = ads_detected.get(model, 0)
        data.append({'Model': model, 'Avg Speed (s)': speed, 'Avg Accuracy': accuracy, 'Ads Detected': ads})

    df = pd.DataFrame(data)

    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=df, x='Avg Speed (s)', y='Avg Accuracy', size='Ads Detected', hue='Model', sizes=(100, 1000), palette='deep', alpha=0.7)
    for i in range(df.shape[0]):
        plt.text(
            df['Avg Speed (s)'][i] + 0.02,
            df['Avg Accuracy'][i] + 0.002,
            df['Model'][i],
            fontsize=10,
            weight='bold',
            verticalalignment='bottom'
        )
    plt.xlabel("Average Transcription Time (s)", fontsize=12)
    plt.ylabel("Average Accuracy (1 - WER)", fontsize=12)
    plt.title("Transcription Accuracy vs. Ads Detected", fontsize=14)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "transcription_accuracy_vs_ads_detected.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Combined metrics scatter plot saved as '{plot_path}'.")

def plot_interactive_scatter(aggregated: dict, run_dir: str):
    """
    Create an interactive scatter plot using Plotly.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=aggregated["avg_speed"],
        y=aggregated["avg_accuracy"],
        mode='markers+text',
        text=aggregated["model"],
        textposition='top center',
        marker=dict(
            size=12,
            color='rgba(135, 206, 250, 0.6)',
            line=dict(width=2, color='DarkSlateGrey')
        )
    ))
    fig.update_layout(
        title='Whisper Models: Interactive Speed vs. Accuracy',
        xaxis_title='Average Transcription Time (s)',
        yaxis_title='Average Accuracy (1 - WER)',
        hovermode='closest',
        template='plotly_white'
    )
    plot_path = os.path.join(run_dir, "whisper_models_interactive_scatter.html")
    fig.write_html(plot_path)
    fig.show()
    logger.info(f"Interactive scatter plot created and saved as '{plot_path}'.")

def plot_heatmap(aggregated: dict, run_dir: str):
    """
    Plot a heatmap of correlation between speed and accuracy for each model.
    """
    import seaborn as sns
    import pandas as pd

    data = []
    for model, speed, accuracy in zip(aggregated["model"], aggregated["avg_speed"], aggregated["avg_accuracy"]):
        data.append({'model': model, 'speed': speed, 'accuracy': accuracy})

    df = pd.DataFrame(data)
    if df.empty:
        logger.warning("No data available for heatmap.")
        return

    correlation = df[['speed', 'accuracy']].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1
    )
    plt.title('Correlation between Speed and Accuracy', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "whisper_models_heatmap_correlation.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Heatmap of correlations saved as '{plot_path}'.")

def plot_ad_detection_precision_recall(ad_detection_metrics: Dict[str, Dict[str, float]], run_dir: str):
    """
    Plot precision, recall, and F1 score for ad detection.
    """
    models = list(ad_detection_metrics.keys())
    precision = [ad_detection_metrics[model]['precision'] for model in models]
    recall = [ad_detection_metrics[model]['recall'] for model in models]
    f1_score = [ad_detection_metrics[model]['f1_score'] for model in models]

    df = pd.DataFrame({
        'Model': models,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    })

    df_melted = df.melt(id_vars='Model', value_vars=['Precision', 'Recall', 'F1 Score'], var_name='Metric', value_name='Score')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric')
    plt.title('Ad Detection Metrics per Model')
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "ad_detection_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Ad detection metrics plot saved as '{plot_path}'.")

def plot_processed_transcription_wer(processed_transcription_metrics: Dict[str, Dict[str, float]], run_dir: str):
    """
    Plot the average WER after ad removal per model.
    """
    models = list(processed_transcription_metrics.keys())
    avg_wer = [processed_transcription_metrics[model]['avg_wer'] for model in models]

    df = pd.DataFrame({
        'Model': models,
        'WER': avg_wer
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Model', y='WER')
    plt.title('Average WER after Ad Removal per Model')
    plt.ylim(0, max(avg_wer) + 0.1)
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "processed_transcription_wer.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Processed transcription WER plot saved as '{plot_path}'.")
