# podcast_processor/reporting/report.py

import os
import logging
import re
import json
from jinja2 import Environment, FileSystemLoader
from typing import Dict, List
from thefuzz import process, fuzz
from podcast_processor.transcription.utils import normalize_text
from podcast_processor.config import TRANSCRIPTIONS_DIR

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

def convert_time_to_seconds(time_str: str) -> float:
    """
    Convert a time string in 'HH:MM:SS.ss', 'MM:SS.ss', or 'SS.ss' format to seconds.
    """
    try:
        parts = time_str.strip().split(':')
        # Ensure that seconds may include fractions
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours = '0'
            minutes, seconds = parts
        elif len(parts) == 1:
            hours = '0'
            minutes = '0'
            seconds = parts[0]
        else:
            return 0.0
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        return total_seconds
    except Exception as e:
        logger.error(f"Error converting time '{time_str}' to seconds: {e}")
        return 0.0

def highlight_ads(transcript_segments: List[Dict], ads: List[Dict]) -> str:
    transcript_text = ''.join([segment['text'] for segment in transcript_segments])
    highlighted_transcript = transcript_text
    offset = 0  # Offset due to added HTML tags

    for ad in ads:
        ad_text = ad.get('text', '')
        if ad_text:
            # Normalize texts
            transcript_text_norm = normalize_text(transcript_text)
            ad_text_norm = normalize_text(ad_text)

            # Find the best match
            match = re.search(re.escape(ad_text_norm), transcript_text_norm)
            if match:
                start_idx = match.start() + offset
                end_idx = match.end() + offset
                # Insert highlight tags
                highlighted_transcript = (
                        highlighted_transcript[:start_idx] +
                        '<span class="ad-highlight">' +
                        highlighted_transcript[start_idx:end_idx] +
                        '</span>' +
                        highlighted_transcript[end_idx:]
                )
                offset += len('<span class="ad-highlight"></span>')
            else:
                logger.warning(f"Ad text not found in transcript: '{ad_text}'")
    return highlighted_transcript




def generate_full_transcript_html(transcript_segments: List[Dict], ads: List[Dict], model_name: str, audio_file: str, run_dir: str):
    """
    Generate and save an HTML file displaying the full transcript with detected ads highlighted.
    """
    # Highlight detected ads in the transcript
    transcript_html = highlight_ads(transcript_segments, ads)

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

def load_transcript_segments(model_name: str, audio_file: str) -> List[Dict]:
    """
    Load the transcript segments from a JSON file.
    """
    transcript_file = os.path.splitext(audio_file)[0] + "_segments.json"
    transcript_path = os.path.join(TRANSCRIPTIONS_DIR, model_name, transcript_file)
    if not os.path.exists(transcript_path):
        logger.error(f"Transcript segments file not found: {transcript_path}")
        return []
    with open(transcript_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    return segments

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
