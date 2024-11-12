# podcast_processor/ad_detection/detect_ads.py

import os
import time
import logging
import openai
from dotenv import load_dotenv
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading  # Import threading for thread names

from openai import RateLimitError, APIConnectionError, OpenAIError, OpenAI

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
load_dotenv()  # Assumes .env is in the root directory
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

def detect_ads(model_name: str, transcription: str, client: OpenAI) -> List[Dict]:
    """
    Detect advertisements in the transcription using the specified LLM model.
    Returns a list of advertisements with their text and approximate positions.
    """
    ads = []
    prompt = (
        "The following is a podcast transcription that contains some advertisements. "
        "Please identify and list the advertisements in this text.\n\n"
        f"{transcription}\n\n"
        "For each advertisement, provide the text and the start and end timestamps in the format:\n"
        "Ad 1:\n"
        "Text: [Ad text]\n"
        "Start: [start_time]\n"
        "End: [end_time]\n\n"
        "If timestamps are not available, approximate the position in the transcript."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that helps identify advertisements in podcast transcriptions.",
                },
                {"role": "user", "content": prompt},
            ],
            #max_tokens=1000,
            #temperature=0.0,
        )
        ads_content = response.choices[0].message.content.strip()
        ads = parse_ads_response(ads_content)
        thread_name = threading.current_thread().name
        logger.info(f"Thread {thread_name}: Detected {len(ads)} ads using model '{model_name}'.")
    except RateLimitError as e:
        thread_name = threading.current_thread().name
        logger.error(f"Thread {thread_name}: Rate limit error for model '{model_name}': {e}")
        time.sleep(60)  # Wait before retrying
    except APIConnectionError as e:
        thread_name = threading.current_thread().name
        logger.error(f"Thread {thread_name}: API connection error for model '{model_name}': {e}")
        time.sleep(10)
    except OpenAIError as e:
        thread_name = threading.current_thread().name
        logger.error(f"Thread {thread_name}: OpenAI error for model '{model_name}': {e}")
    except Exception as e:
        thread_name = threading.current_thread().name
        logger.error(f"Thread {thread_name}: Unexpected error detecting ads with model '{model_name}': {e}")

    return ads

def detect_ads_for_model(model_name: str, run_dir: str, client: OpenAI) -> Dict[str, List[Dict]]:
    """
    Detect advertisements in all transcriptions using a specific LLM model.
    Returns a dictionary with audio file names as keys and lists of detected ads as values.
    """
    results = {}
    model_transcriptions_dir = TRANSCRIPTIONS_DIR

    for whisper_model in os.listdir(model_transcriptions_dir):
        whisper_model_dir = os.path.join(model_transcriptions_dir, whisper_model)
        if not os.path.isdir(whisper_model_dir):
            continue
        for transcription_file in os.listdir(whisper_model_dir):
            if not transcription_file.endswith('.txt'):
                continue
            audio_file = os.path.splitext(transcription_file)[0] + ".mp3"  # Adjust if different extensions
            transcription_path = os.path.join(whisper_model_dir, transcription_file)
            transcription_text = read_transcription(transcription_path)
            if not transcription_text:
                thread_name = threading.current_thread().name
                logger.warning(f"Thread {thread_name}: Empty transcription for '{audio_file}'. Skipping ad detection.")
                continue
            ads = detect_ads(model_name, transcription_text, client)
            results[audio_file] = ads
            save_ad_detections(ads, model_name, audio_file, AD_DETECTIONS_DIR)

    return results

def detect_ads_in_transcriptions(run_dir: str) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Detect advertisements in all transcriptions using all specified LLM models.
    Returns a nested dictionary with model names, audio files, and detected ads.
    """
    llm_models, client = get_llm_models()
    ad_detections = {model: {} for model in llm_models}

    logger.info(f"Starting ad detection with {len(llm_models)} parallel threads.")

    with ThreadPoolExecutor(max_workers=len(llm_models)) as executor:
        future_to_model = {executor.submit(detect_ads_for_model, model, run_dir, client): model for model in llm_models}

        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                detections = future.result()
                ad_detections[model_name] = detections
            except Exception as e:
                logger.exception(f"Error detecting ads with model {model_name}: {e}")

    return ad_detections
