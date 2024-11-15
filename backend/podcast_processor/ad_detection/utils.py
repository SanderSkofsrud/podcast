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
        response_text = re.sub(r'^```(json)?', '', response_text)
        response_text = response_text.strip('`')

        # Use regular expressions to extract the JSON array
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if match:
            json_text = match.group(0)
            ads = json.loads(json_text)
            # Ensure 'start' and 'end' are floats
            for ad in ads:
                ad['start'] = float(ad.get('start', 0))
                ad['end'] = float(ad.get('end', 0))
                ad['text'] = ad.get('text', '').strip()
            return ads
        else:
            logger.error("No JSON array found in the response.")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing ads response: {e}")
        return []


def save_ad_detections(ads: List[Dict], whisper_model: str, llm_model: str, audio_file: str, ad_detections_dir: str):
    """
    Save the detected ads with their timestamps to a JSON file.
    """
    try:
        model_dir = os.path.join(ad_detections_dir, whisper_model, llm_model)
        os.makedirs(model_dir, exist_ok=True)
        ad_detection_file = os.path.splitext(audio_file)[0] + "_ads.json"
        ad_detection_path = os.path.join(model_dir, ad_detection_file)
        with open(ad_detection_path, 'w', encoding='utf-8') as f:
            json.dump(ads, f, indent=4)
        logger.info(f"Saved ad detections to '{ad_detection_path}'.")
    except Exception as e:
        logger.error(f"Error saving ad detections for '{audio_file}' with models '{whisper_model}' and '{llm_model}': {e}")

