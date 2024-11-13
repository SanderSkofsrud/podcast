# podcast_processor/reporting/report.py

import os
import logging
import difflib
from jinja2 import Environment, FileSystemLoader, Template
from typing import Dict, List

logger = logging.getLogger(__name__)

def generate_diff_html(reference: str, hypothesis: str, model_name: str, audio_file: str, run_dir: str):
    """
    Generate and save an HTML file highlighting differences between normalized reference and hypothesis.
    """
    from podcast_processor.reporting.html_utils import get_diff_html
    diff_html = get_diff_html(reference, hypothesis)

    # Create a Jinja2 environment pointing to the templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("diff_template.html")

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
    highlighted_transcript = transcript

    for ad in ads:
        # Find approximate matches of the ad text in the transcript
        s = difflib.SequenceMatcher(None, transcript.lower(), ad.lower())
        match = s.find_longest_match(0, len(transcript), 0, len(ad))
        if match.size > 0:
            # Extract the matching text from the transcript
            match_text = transcript[match.a: match.a + match.size]
            # Highlight the matching text
            highlighted_transcript = highlighted_transcript.replace(
                match_text,
                f'<span class="ad-highlight">{match_text}</span>'
            )
    return highlighted_transcript


def generate_full_transcript_html(transcript: str, ads: List[str], model_name: str, audio_file: str, run_dir: str):
    """
    Generate and save an HTML file displaying the full transcript with detected ads highlighted.
    """
    # Highlight detected ads in the transcript
    transcript_html = highlight_ads(transcript, ads)

    # Create a Jinja2 environment pointing to the templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("full_transcript_template.html")

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

def generate_summary_report(aggregated: dict, ad_detections: Dict[str, Dict[str, List[Dict]]], ad_detection_metrics: Dict[str, Dict[str, float]], processed_transcription_metrics: Dict[str, Dict[str, float]], run_dir: str):
    """
    Generate a summary HTML report linking all visualizations, transcription diffs, and full transcripts with ad highlights.
    """
    # Gather all audio files from one of the models
    audio_files = []
    for model in aggregated["model"]:
        model_dir = os.path.join(run_dir, "transcription_diffs", model)
        if os.path.exists(model_dir):
            audio_files = [os.path.splitext(f)[0] for f in os.listdir(model_dir) if f.endswith('.html')]
            break

    # Create a Jinja2 environment pointing to the templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("summary_template.html")

    rendered_html = template.render(
        run_dir=run_dir,
        aggregated=aggregated,
        ad_detections=ad_detections,
        ad_detection_metrics=ad_detection_metrics,
        processed_transcription_metrics=processed_transcription_metrics,
        audio_files=audio_files
    )

    report_path = os.path.join(run_dir, "summary_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(rendered_html)

    logger.info(f"Summary report generated at '{report_path}'.")
