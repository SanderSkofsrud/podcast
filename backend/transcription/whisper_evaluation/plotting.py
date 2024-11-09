# whisper_evaluation/plotting.py
import difflib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template
from whisper_evaluation.config import WHISPER_MODELS

logger = logging.getLogger(__name__)

def plot_scatter(aggregated: dict, run_dir: str):
    """
    Plot an enhanced scatter plot for speed vs. accuracy.

    Args:
        aggregated (dict): Dictionary containing lists of 'model', 'avg_speed', and 'avg_accuracy'.
        run_dir (str): Directory where the plot will be saved.
    """
    plt.figure(figsize=(12, 8))

    # Create scatter plot with varying sizes or colors if needed
    scatter = plt.scatter(
        aggregated["avg_speed"],
        aggregated["avg_accuracy"],
        color='skyblue',
        s=100,  # Increase point size for better visibility
        edgecolors='k'  # Add black edge for contrast
    )

    # Annotate each point with the model name
    for i, model in enumerate(aggregated["model"]):
        plt.text(
            aggregated["avg_speed"][i] + 0.02,  # Slight offset for readability
            aggregated["avg_accuracy"][i] + 0.002,  # Slight offset for readability
            model,
            fontsize=10,
            weight='bold',
            verticalalignment='bottom'
        )

    # Set plot labels and title
    plt.xlabel("Average Transcription Time (s)", fontsize=12)
    plt.ylabel("Average Accuracy (1 - WER)", fontsize=12)
    plt.title("Whisper Models: Speed vs. Accuracy", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save and display the plot
    plot_path = os.path.join(run_dir, "whisper_models_speed_vs_accuracy_enhanced.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Enhanced scatter plot saved as '{plot_path}'.")

def plot_bar_charts(aggregated: dict, run_dir: str):
    """
    Plot bar charts for average speed and accuracy of each Whisper model.

    Args:
        aggregated (dict): Dictionary containing lists of 'model', 'avg_speed', and 'avg_accuracy'.
        run_dir (str): Directory where the plot will be saved.
    """
    models = aggregated["model"]
    avg_speed = aggregated["avg_speed"]
    avg_accuracy = aggregated["avg_accuracy"]

    x = range(len(models))

    fig, ax1 = plt.subplots(figsize=(14, 8))

    color = 'tab:blue'
    ax1.set_xlabel('Whisper Models', fontsize=12)
    ax1.set_ylabel('Average Transcription Time (s)', color=color, fontsize=12)
    bars1 = ax1.bar(x, avg_speed, color=color, alpha=0.6, label='Avg Speed')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2.set_ylabel('Average Accuracy (1 - WER)', color=color, fontsize=12)
    bars2 = ax2.bar([i + 0.2 for i in x], avg_accuracy, color=color, alpha=0.6, width=0.4, label='Avg Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    plt.title('Whisper Models: Average Speed and Accuracy', fontsize=14)
    plt.tight_layout()

    # Save and display the plot
    plot_path = os.path.join(run_dir, "whisper_models_bar_charts.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Bar charts saved as '{plot_path}'.")

def plot_box_plots(results: dict, run_dir: str):
    """
    Plot box plots for speed and accuracy distributions of each Whisper model.

    Args:
        results (dict): Dictionary with model names as keys and dictionaries containing
                       'speed' and 'accuracy' lists as values.
        run_dir (str): Directory where the plot will be saved.
    """
    models = list(results.keys())
    speed_data = [results[model]["speed"] for model in models]
    accuracy_data = [results[model]["accuracy"] for model in models]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Box plot for Speed
    axes[0].boxplot(speed_data, labels=models, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red'))
    axes[0].set_title('Transcription Speed Distribution', fontsize=14)
    axes[0].set_xlabel('Whisper Models', fontsize=12)
    axes[0].set_ylabel('Transcription Time (s)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # Box plot for Accuracy
    axes[1].boxplot(accuracy_data, labels=models, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', color='green'),
                    medianprops=dict(color='red'))
    axes[1].set_title('Accuracy Distribution (1 - WER)', fontsize=14)
    axes[1].set_xlabel('Whisper Models', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save and display the plot
    plot_path = os.path.join(run_dir, "whisper_models_box_plots.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Box plots saved as '{plot_path}'.")

def plot_combined_performance(aggregated: dict, run_dir: str):
    """
    Plot combined performance metrics: Speed vs. Accuracy with model size indication.

    Args:
        aggregated (dict): Dictionary containing lists of 'model', 'avg_speed', and 'avg_accuracy'.
        run_dir (str): Directory where the plot will be saved.
    """
    plt.figure(figsize=(12, 8))

    # Assume model size increases with the order in WHISPER_MODELS
    sizes = [WHISPER_MODELS.index(model) + 1 for model in aggregated["model"]]

    scatter = plt.scatter(
        aggregated["avg_speed"],
        aggregated["avg_accuracy"],
        s=[size * 100 for size in sizes],  # Scale sizes for visibility
        alpha=0.6,
        c=sizes,  # Color based on size
        cmap='viridis',
        edgecolors='k'
    )

    # Annotate each point with the model name
    for i, model in enumerate(aggregated["model"]):
        plt.text(
            aggregated["avg_speed"][i] + 0.02,
            aggregated["avg_accuracy"][i] + 0.002,
            model,
            fontsize=10,
            weight='bold',
            verticalalignment='bottom'
        )

    # Add colorbar to indicate model size
    cbar = plt.colorbar(scatter)
    cbar.set_label('Model Size (1=Tiny, 5=Large)', fontsize=12)

    plt.xlabel("Average Transcription Time (s)", fontsize=12)
    plt.ylabel("Average Accuracy (1 - WER)", fontsize=12)
    plt.title("Whisper Models: Combined Speed, Accuracy, and Size", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save and display the plot
    plot_path = os.path.join(run_dir, "whisper_models_combined_performance.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Combined performance plot saved as '{plot_path}'.")

def plot_tradeoff_frontier(aggregated: dict, run_dir: str):
    """
    Plot the trade-off frontier to identify Pareto-efficient models.

    Args:
        aggregated (dict): Dictionary containing lists of 'model', 'avg_speed', and 'avg_accuracy'.
        run_dir (str): Directory where the plot will be saved.
    """
    models = aggregated["model"]
    avg_speed = aggregated["avg_speed"]
    avg_accuracy = aggregated["avg_accuracy"]

    # Sort models by speed
    sorted_indices = sorted(range(len(avg_speed)), key=lambda i: avg_speed[i])
    sorted_speed = [avg_speed[i] for i in sorted_indices]
    sorted_accuracy = [avg_accuracy[i] for i in sorted_indices]
    sorted_models = [models[i] for i in sorted_indices]

    plt.figure(figsize=(12, 8))
    plt.plot(sorted_speed, sorted_accuracy, marker='o', linestyle='-', color='purple')

    for i, model in enumerate(sorted_models):
        plt.text(
            sorted_speed[i] + 0.02,
            sorted_accuracy[i] + 0.002,
            model,
            fontsize=10,
            weight='bold',
            verticalalignment='bottom'
        )

    plt.xlabel("Average Transcription Time (s)", fontsize=12)
    plt.ylabel("Average Accuracy (1 - WER)", fontsize=12)
    plt.title("Whisper Models: Performance Trade-off Frontier", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save and display the plot
    plot_path = os.path.join(run_dir, "whisper_models_tradeoff_frontier.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Trade-off frontier plot saved as '{plot_path}'.")

def plot_interactive_scatter(aggregated: dict, run_dir: str):
    """
    Create an interactive scatter plot using Plotly.

    Args:
        aggregated (dict): Dictionary containing lists of 'model', 'avg_speed', and 'avg_accuracy'.
        run_dir (str): Directory where the plot will be saved.
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

    # Save and display the interactive plot
    plot_path = os.path.join(run_dir, "whisper_models_interactive_scatter.html")
    fig.write_html(plot_path)
    fig.show()
    logger.info(f"Interactive scatter plot created and saved as '{plot_path}'.")

def plot_heatmap(results: dict, run_dir: str):
    """
    Plot a heatmap of correlation between speed and accuracy for each model.

    Args:
        results (dict): Dictionary with model names as keys and dictionaries containing
                       'speed' and 'accuracy' lists as values.
        run_dir (str): Directory where the plot will be saved.
    """
    data = []
    for model, metrics in results.items():
        for speed, accuracy in zip(metrics["speed"], metrics["accuracy"]):
            data.append({'model': model, 'speed': speed, 'accuracy': accuracy})

    df = pd.DataFrame(data)
    if df.empty:
        logger.warning("No data available for heatmap.")
        return

    # Calculate correlation between speed and accuracy for each model
    correlation = {}
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        if len(model_df) > 1:
            corr = model_df['speed'].corr(model_df['accuracy'])
            correlation[model] = corr
        else:
            correlation[model] = float('nan')  # Not enough data to calculate correlation

    # Convert to DataFrame for heatmap
    corr_df = pd.DataFrame.from_dict(correlation, orient='index', columns=['Correlation']).dropna()

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        corr_df.T,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1
    )
    plt.title('Correlation between Speed and Accuracy per Model', fontsize=14)
    plt.tight_layout()

    # Save and display the plot
    plot_path = os.path.join(run_dir, "whisper_models_heatmap_correlation.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Heatmap of correlations saved as '{plot_path}'.")

def get_diff_html(reference: str, hypothesis: str) -> str:
    """
    Generate an HTML representation highlighting differences between reference and hypothesis.

    Args:
        reference (str): The ground truth transcription.
        hypothesis (str): The model's transcription.

    Returns:
        str: HTML string with differences highlighted.
    """
    # Tokenize the texts into words
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    # Initialize the HTML output
    html_output = []
    html_output.append("<p>")

    # Use difflib to get the differences
    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            html_output.append(' '.join(ref_tokens[i1:i2]))
        elif tag == 'replace':
            html_output.append('<span style="background-color: #ff9999;">' + ' '.join(hyp_tokens[j1:j2]) + '</span>')
        elif tag == 'insert':
            html_output.append('<span style="background-color: #ffcc99;">' + ' '.join(hyp_tokens[j1:j2]) + '</span>')
        elif tag == 'delete':
            html_output.append('<span style="background-color: #99ccff;">' + ' '.join(ref_tokens[i1:i2]) + '</span>')
        html_output.append(' ')  # Add space between words

    html_output.append("</p>")

    return ''.join(html_output)

def generate_diff_html(reference: str, hypothesis: str, model_name: str, audio_file: str, run_dir: str):
    """
    Generate and save an HTML file highlighting differences between reference and hypothesis.

    Args:
        reference (str): The ground truth transcription.
        hypothesis (str): The model's transcription.
        model_name (str): Name of the Whisper model.
        audio_file (str): Name of the audio file.
        run_dir (str): Directory where the HTML file will be saved.
    """
    diff_html = get_diff_html(reference, hypothesis)

    # Create a Jinja2 template for better styling (optional)
    template = Template("""
    <html>
    <head>
        <title>Transcription Comparison - {{ model_name }} - {{ audio_file }}</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { font-size: 20px; margin-bottom: 10px; }
            .transcription { font-size: 16px; }
            .replace { background-color: #ff9999; }
            .insert { background-color: #ffcc99; }
            .delete { background-color: #99ccff; }
        </style>
    </head>
    <body>
        <div class="header">Model: {{ model_name }} | Audio File: {{ audio_file }}</div>
        <div class="transcription">
            {{ diff_html | safe }}
        </div>
    </body>
    </html>
    """)

    rendered_html = template.render(
        model_name=model_name,
        audio_file=audio_file,
        diff_html=diff_html
    )

    # Define the path for the HTML file
    model_dir = os.path.join(run_dir, "transcription_diffs", model_name)
    os.makedirs(model_dir, exist_ok=True)
    html_filename = os.path.splitext(audio_file)[0] + ".html"
    html_path = os.path.join(model_dir, html_filename)

    # Save the HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(rendered_html)

    logger.info(f"Saved transcription diff HTML to '{html_path}'.")

def generate_summary_report(aggregated: dict, run_dir: str):
    """
    Generate a summary HTML report linking all visualizations and transcription diffs.

    Args:
        aggregated (dict): Aggregated metrics.
        run_dir (str): Directory where the report will be saved.
    """
    # Gather all audio files from one of the models
    audio_files = []
    for model in aggregated["model"]:
        model_dir = os.path.join(run_dir, "transcription_diffs", model)
        if os.path.exists(model_dir):
            audio_files = [f for f in os.listdir(model_dir) if f.endswith('.html')]
            break  # Assuming all models have the same audio files

    # Jinja2 template for the summary report
    template = Template("""
    <html>
    <head>
        <title>Model Evaluation Summary - {{ run_dir }}</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { text-align: center; }
            .section { margin-bottom: 40px; }
            .section img { max-width: 100%; height: auto; }
            .transcription-link { margin-right: 10px; }
        </style>
    </head>
    <body>
        <h1>Model Evaluation Summary</h1>
        <div class="section">
            <h2>Aggregate Metrics</h2>
            <p>Average Speed and Accuracy for each model are visualized in the following plots:</p>
            <ul>
                <li><a href="whisper_models_speed_vs_accuracy_enhanced.png">Speed vs. Accuracy Scatter Plot</a></li>
                <li><a href="whisper_models_bar_charts.png">Bar Charts</a></li>
                <li><a href="whisper_models_box_plots.png">Box Plots</a></li>
                <li><a href="whisper_models_combined_performance.png">Combined Performance</a></li>
                <li><a href="whisper_models_tradeoff_frontier.png">Trade-off Frontier</a></li>
                <li><a href="whisper_models_interactive_scatter.html">Interactive Scatter Plot</a></li>
                <li><a href="whisper_models_heatmap_correlation.png">Heatmap of Correlations</a></li>
            </ul>
        </div>
        <div class="section">
            <h2>Transcription Differences</h2>
            <p>Click on a model to view the transcription differences for each audio file:</p>
            <ul>
                {% for model in aggregated['model'] %}
                <li>
                    <strong>{{ model }}</strong>
                    <ul>
                        {% for audio_file in audio_files %}
                        <li><a href="transcription_diffs/{{ model }}/{{ audio_file }}.html">{{ audio_file }}</a></li>
                        {% endfor %}
                    </ul>
                </li>
                {% endfor %}
            </ul>
        </div>
    </body>
    </html>
    """)

    rendered_html = template.render(
        run_dir=run_dir,
        aggregated=aggregated,
        audio_files=[os.path.splitext(f)[0] for f in audio_files]
    )

    report_path = os.path.join(run_dir, "summary_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(rendered_html)

    logger.info(f"Summary report generated at '{report_path}'.")
