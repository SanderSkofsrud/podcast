# podcast_processor/ad_detection/utils.py

import os
import re
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def parse_ads_response(response_text: str) -> List[Dict]:
    try:
        response_text = response_text.strip()

        # Remove code block markers and language identifiers
        if response_text.startswith("```") and response_text.endswith("```"):
            response_text = response_text[3:-3].strip()
            if response_text.startswith('json'):
                response_text = response_text[4:].strip()

        ads = json.loads(response_text)
        # Ensure 'start' and 'end' are converted to floats
        for ad in ads:
            ad['start'] = float(ad['start'])
            ad['end'] = float(ad['end'])
            ad['text'] = ad['text'].strip()
        return ads
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing ads response: {e}")
        return []

def save_ad_detections(ads: List[Dict], model_name: str, audio_file: str, ad_detections_dir: str):
    """
    Save the detected ads with their timestamps to a JSON file.
    """
    try:
        model_dir = os.path.join(ad_detections_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        ad_detection_file = os.path.splitext(audio_file)[0] + "_ads.json"
        ad_detection_path = os.path.join(model_dir, ad_detection_file)
        with open(ad_detection_path, 'w', encoding='utf-8') as f:
            json.dump(ads, f, indent=4)
        logger.info(f"Saved ad detections to '{ad_detection_path}'.")
    except Exception as e:
        logger.error(f"Error saving ad detections for '{audio_file}' with model '{model_name}': {e}")
