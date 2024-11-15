# podcast_processor/ad_detection/detect_ads.py
import json
import os
import time
import logging
import openai
from dotenv import load_dotenv
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from openai import RateLimitError, APIConnectionError, OpenAIError

from podcast_processor.config import (
    AD_DETECTIONS_DIR,
    LLM_MODELS,
    TRANSCRIPTIONS_DIR
)
from podcast_processor.ad_detection.models import get_llm_models
from podcast_processor.ad_detection.utils import parse_ads_response, save_ad_detections
from podcast_processor.transcription.utils import normalize_text

logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def read_transcription(file_path: str) -> str:
    """
    Read the transcription text from a file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading transcription from '{file_path}': {e}")
        return ""

def detect_ads(model_name: str, transcription_segments: List[Dict]) -> List[Dict]:
    """
    Detect advertisements in a transcription using the specified LLM model.
    """
    ads = []
    transcription_text = ''
    for segment in transcription_segments:
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        text = segment.get('text', '').strip()
        transcription_text += f"[{start:.2f} - {end:.2f}] {text}\n"

    prompt = (
        "You will be provided with a podcast transcription that contains some advertisements. "
        "Each line of the transcription includes the start and end times in seconds, followed by the text.\n\n"
        f"Transcription:\n{transcription_text}\n\n"
        "Please identify and list the advertisements in the transcription.\n\n"
        "For each advertisement, provide the following details in JSON format:\n"
        "[\n"
        "  {{\n"
        "    \"text\": \"Ad text\",\n"
        "    \"start\": Start time in seconds,\n"
        "    \"end\": End time in seconds\n"
        "  }},\n"
        "  ...\n"
        "]\n\n"
        "Ensure that the response is a valid JSON array without any code blocks or additional text."
    )
    max_retries = 5
    retry_delay = 10  # Start with 10 seconds

    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that helps identify advertisements in podcast transcriptions.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            ads_content = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM response for model '{model_name}':\n{ads_content}")
            ads = parse_ads_response(ads_content)
            thread_name = threading.current_thread().name
            logger.info(f"Thread {thread_name}: Detected {len(ads)} ads using model '{model_name}'.")

            # Handle ads with missing 'start' or 'end' times
            for ad in ads:
                start = ad.get('start', '-1')
                end = ad.get('end', '-1')
                try:
                    ad['start'] = int(float(start)) if start != '-1' else None
                    ad['end'] = int(float(end)) if end != '-1' else None
                except (ValueError, TypeError):
                    logger.warning(f"Invalid 'start' or 'end' time in ad: {ad}")
                    ad['start'] = None
                    ad['end'] = None

            return ads  # Success, return the ads
        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break  # Exit on other exceptions
    return ads  # Return any ads detected before failure

def detect_ads_for_model(llm_model_name: str, run_dir: str):
    """
    Detect advertisements in all transcriptions using a specific LLM model.
    """
    results = {}
    transcription_base_dir = TRANSCRIPTIONS_DIR  # This is the 'data/transcriptions' directory

    # Iterate over each Whisper model directory (e.g., 'tiny')
    for whisper_model in os.listdir(transcription_base_dir):
        whisper_model_dir = os.path.join(transcription_base_dir, whisper_model)
        if not os.path.isdir(whisper_model_dir):
            continue

        # Iterate over each transcription file in the Whisper model directory
        for transcription_file in os.listdir(whisper_model_dir):
            if not transcription_file.endswith('_segments.json'):
                continue
            audio_file = transcription_file.replace('_segments.json', '.mp3')
            transcription_path = os.path.join(whisper_model_dir, transcription_file)
            transcription_segments = load_transcription_segments(transcription_path)
            if not transcription_segments:
                thread_name = threading.current_thread().name
                logger.warning(f"Thread {thread_name}: Empty transcription segments for '{audio_file}'. Skipping ad detection.")
                continue
            # Process the transcription segments using the LLM model
            ads = detect_ads(llm_model_name, transcription_segments)
            results[audio_file] = ads
            # Save the detected ads, including the whisper_model
            save_ad_detections(ads, whisper_model, llm_model_name, audio_file, AD_DETECTIONS_DIR)

    return results


def detect_ads_in_transcriptions(run_dir: str) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Detect advertisements in all transcriptions using all specified LLM models.
    Returns a nested dictionary with model names, audio files, and detected ads.
    """
    llm_models = LLM_MODELS  # List of LLM models
    ad_detections = {}

    logger.info(f"Starting ad detection with {len(llm_models)} parallel threads.")

    with ThreadPoolExecutor(max_workers=len(llm_models)) as executor:
        future_to_model = {executor.submit(detect_ads_for_model, model, run_dir): model for model in llm_models}

        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                detections = future.result()
                ad_detections[model_name] = detections
            except Exception as e:
                logger.exception(f"Error detecting ads with model {model_name}: {e}")

    return ad_detections

def load_transcription_segments(file_path: str) -> List[Dict]:
    """
    Load the transcription segments from a JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
            return segments
    except Exception as e:
        logger.error(f"Error loading transcription segments from '{file_path}': {e}")
        return []
