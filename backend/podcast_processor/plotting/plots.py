# podcast_processor/plotting/plots.py

import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List
from podcast_processor.config import WHISPER_MODELS, LLM_MODELS, LLM_MODEL_CONFIGS
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

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

def plot_ad_detection_metrics(ad_detections: Dict[str, Dict[str, Dict[str, List[Dict]]]], run_dir: str):
    """
    Plot advertisement detection metrics such as number of ads detected per model combination.
    """
    model_ads_count = {}
    for whisper_model, llm_models in ad_detections.items():
        for llm_model, detections in llm_models.items():
            model_key = f"{whisper_model}_{llm_model}"
            total_ads = sum(len(ads) for ads in detections.values())
            model_ads_count[model_key] = total_ads

    # Prepare data for plotting
    models = list(model_ads_count.keys())
    ads_counts = [model_ads_count[model] for model in models]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=models, y=ads_counts, palette="viridis")
    plt.xlabel("Model Combinations", fontsize=12)
    plt.ylabel("Total Advertisements Detected", fontsize=12)
    plt.title("Advertisements Detected per Model Combination", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "ad_detections_per_model.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Advertisement detection bar plot saved as '{plot_path}'.")


def plot_combined_metrics(aggregated: dict, ad_detection_metrics: Dict[str, Dict[str, float]], run_dir: str):
    """
    Create combined plots that show transcription accuracy and ad detection F1 score together.
    """
    # Extract F1 scores per model combination
    f1_scores = {model: metrics.get('f1_score', 0.0) for model, metrics in ad_detection_metrics.items()}

    # Create a DataFrame for plotting
    data = []
    for model in aggregated["model"]:
        speed = aggregated["avg_speed"][aggregated["model"].index(model)]
        accuracy = aggregated["avg_accuracy"][aggregated["model"].index(model)]
        f1_score = f1_scores.get(model, 0.0)
        data.append({'Model': model, 'Avg Speed (s)': speed, 'Avg Accuracy': accuracy, 'F1 Score': f1_score})

    df = pd.DataFrame(data)

    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=df,
        x='Avg Speed (s)',
        y='Avg Accuracy',
        size='F1 Score',
        hue='Model',
        sizes=(100, 1000),
        palette='deep',
        alpha=0.7
    )
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
    plt.title("Transcription Accuracy vs. Ad Detection F1 Score", fontsize=14)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "transcription_accuracy_vs_ad_detection_f1_score.png")
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

def plot_confusion_matrix(y_true, y_pred, model_name, run_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Ad', 'Ad'], yticklabels=['Non-Ad', 'Ad'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.tight_layout()
    plot_path = os.path.join(run_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Confusion matrix saved as '{plot_path}'.")

def plot_roc_curve(y_true, y_scores, model_name, run_dir):
    if len(set(y_scores)) <= 1:
        logger.warning(f"Cannot plot ROC curve for '{model_name}' as y_scores contain a single class.")
        return
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plot_path = os.path.join(run_dir, f"roc_curve_{model_name}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"ROC curve saved as '{plot_path}'.")

def plot_precision_recall_curve(y_true, y_scores, model_name, run_dir):
    if len(set(y_scores)) <= 1:
        logger.warning(f"Cannot plot Precision-Recall curve for '{model_name}' as y_scores contain a single class.")
        return
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, lw=2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve - {model_name}')
    plot_path = os.path.join(run_dir, f"precision_recall_curve_{model_name}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Precision-Recall curve saved as '{plot_path}'.")

def plot_ad_positions(ad_detections, run_dir):
    for whisper_model, llm_models in ad_detections.items():
        for llm_model_name, detections in llm_models.items():
            model_name = f"{whisper_model}_{llm_model_name}"
            for audio_file, ads in detections.items():
                if not ads:
                    continue  # Skip if there are no ads
                durations = [ad['end'] - ad['start'] for ad in ads]
                positions = [(ad['start'] + ad['end']) / 2 for ad in ads]
                plt.figure()
                plt.scatter(positions, durations)
                plt.xlabel('Time (s)')
                plt.ylabel('Ad Duration (s)')
                plt.title(f'Ad Positions for {audio_file} using {model_name}')
                plot_filename = f"ad_positions_{model_name}_{audio_file.replace('.mp3', '')}.png"
                plot_path = os.path.join(run_dir, plot_filename)
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Ad positions plot saved as '{plot_path}'.")


def plot_price_vs_ad_accuracy(ad_detection_metrics: dict, run_dir: str):
    """
    Plot Price vs. Ad Detection Accuracy (F1 Score) for each LLM model.
    """
    try:
        data = []
        for model in ad_detection_metrics.keys():
            price = LLM_MODEL_CONFIGS.get(model, {}).get("price", None)
            f1_score = ad_detection_metrics.get(model, {}).get("f1_score", None)
            if price is not None and f1_score is not None:
                data.append({"Model": model, "Price": price, "F1 Score": f1_score})
            else:
                logger.warning(f"Missing data for model '{model}'. Skipping.")

        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="Price", y="F1 Score", hue="Model", s=100, palette="viridis")

        for i in range(df.shape[0]):
            plt.text(
                df["Price"][i] + 0.01,
                df["F1 Score"][i] + 0.001,
                df["Model"][i],
                fontsize=9,
                weight='bold'
            )

        plt.xlabel("Price per 1M Tokens ($)", fontsize=12)
        plt.ylabel("Ad Detection F1 Score", fontsize=12)
        plt.title("Price vs. Ad Detection Accuracy (F1 Score)", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plot_path = os.path.join(run_dir, "price_vs_ad_detection_accuracy.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Price vs. Ad Detection Accuracy plot saved as '{plot_path}'.")
    except Exception as e:
        logger.error(f"Error plotting Price vs. Ad Detection Accuracy: {e}")

def plot_price_vs_time_usage(ad_detection_metrics: dict, run_dir: str):
    """
    Plot Price vs. Time Usage (Processing Time in seconds) for each LLM model.
    """
    try:
        data = []
        for model in ad_detection_metrics.keys():
            price = LLM_MODEL_CONFIGS.get(model, {}).get("price", None)
            processing_time = ad_detection_metrics.get(model, {}).get("processing_time", None)
            if price is not None and processing_time is not None:
                data.append({"Model": model, "Price": price, "Processing Time (s)": processing_time})
            else:
                logger.warning(f"Missing data for model '{model}'. Skipping.")

        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="Price", y="Processing Time (s)", hue="Model", s=100, palette="magma")

        for i in range(df.shape[0]):
            plt.text(
                df["Price"][i] + 0.01,
                df["Processing Time (s)"][i] + 0.1,
                df["Model"][i],
                fontsize=9,
                weight='bold'
            )

        plt.xlabel("Price per 1M Tokens ($)", fontsize=12)
        plt.ylabel("Processing Time (s)", fontsize=12)
        plt.title("Price vs. Time Usage", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plot_path = os.path.join(run_dir, "price_vs_time_usage.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Price vs. Time Usage plot saved as '{plot_path}'.")
    except Exception as e:
        logger.error(f"Error plotting Price vs. Time Usage: {e}")
