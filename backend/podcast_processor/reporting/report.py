# podcast_processor/reporting/report.py

import os
import logging
import difflib
import re
from jinja2 import Environment, FileSystemLoader, Template
from podcast_processor.config import REPORTS_DIR, PLOTS_DIR, AD_DETECTIONS_DIR, LLM_MODELS
from typing import Dict, List
from podcast_processor.transcription.utils import normalize_text  # Ensure this import exists
from podcast_processor.reporting.html_utils import get_diff_html  # Import the utility function

logger = logging.getLogger(__name__)

def generate_diff_html(reference: str, hypothesis: str, model_name: str, audio_file: str, run_dir: str):
    """
    Generate and save an HTML file highlighting differences between normalized reference and hypothesis.
    """
    diff_html = get_diff_html(reference, hypothesis)

    # Create a Jinja2 environment pointing to the templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("diff_template.html")  # Use the template file

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

def highlight_ads(transcript: str, ads: List[str]) -> str:
    """
    Highlight detected advertisements within the transcript by wrapping them in a span with a specific class.

    Args:
        transcript (str): The full transcript text.
        ads (List[str]): A list of detected advertisement phrases.

    Returns:
        str: The HTML-formatted transcript with ads highlighted.
    """
    for ad in sorted(ads, key=lambda x: len(x), reverse=True):  # Sort ads by length to handle overlaps
        # Escape special characters in ad phrases to prevent regex issues
        escaped_ad = re.escape(ad)
        # Use regex to replace exact matches, case-insensitive
        # \b ensures word boundaries to match whole words/phrases
        transcript = re.sub(
            rf'\b({escaped_ad})\b',
            r'<span class="ad-highlight">\1</span>',
            transcript,
            flags=re.IGNORECASE
        )
    return transcript

def generate_full_transcript_html(transcript: str, ads: List[str], model_name: str, audio_file: str, run_dir: str):
    """
    Generate and save an HTML file displaying the full transcript with detected ads highlighted in green.
    """
    # Highlight detected ads in the transcript
    transcript_html = highlight_ads(transcript, ads)

    # Create a Jinja2 environment pointing to the templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("full_transcript_template.html")  # Use the new template file

    rendered_html = template.render(
        model_name=model_name,
        audio_file=audio_file,
        transcript_html=transcript_html
    )

    # Define the path for the HTML file
    transcript_dir = os.path.join(run_dir, "full_transcripts", model_name)
    os.makedirs(transcript_dir, exist_ok=True)
    html_filename = os.path.splitext(audio_file)[0] + ".html"
    html_path = os.path.join(transcript_dir, html_filename)

    # Save the HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(rendered_html)

    logger.info(f"Saved full transcript HTML to '{html_path}'.")

def generate_summary_report(aggregated: dict, ad_detections: Dict[str, Dict[str, List[str]]], run_dir: str):
    """
    Generate a summary HTML report linking all visualizations, transcription diffs, and full transcripts with ad highlights.
    """
    # Gather all audio files from one of the models
    audio_files = []
    for model in aggregated["model"]:
        model_dir = os.path.join(run_dir, "transcription_diffs", model)
        if os.path.exists(model_dir):
            audio_files = [os.path.splitext(f)[0] for f in os.listdir(model_dir) if f.endswith('.html')]
            break  # Assuming all models have the same audio files

    # Create a Jinja2 environment pointing to the templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("summary_template.html")  # Use the template file

    rendered_html = template.render(
        run_dir=run_dir,
        aggregated=aggregated,
        ad_detections=ad_detections,
        audio_files=audio_files
    )

    report_path = os.path.join(run_dir, "summary_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(rendered_html)

    logger.info(f"Summary report generated at '{report_path}'.")
