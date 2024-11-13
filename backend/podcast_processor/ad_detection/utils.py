# podcast_processor/ad_detection/utils.py
import os
import re
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def parse_ads_response(response: str) -> List[Dict]:
    """
    Parse the LLM response to extract advertisements with text and timestamps.
    """
    ads = []
    ad_blocks = re.split(r'Ad\s*\d+:', response, flags=re.IGNORECASE)[1:]
    for block in ad_blocks:
        text_match = re.search(r'Text:\s*(.*?)(?:\n|$)', block, re.DOTALL | re.IGNORECASE)
        start_match = re.search(r'Start:\s*([\d:.]+)', block, re.IGNORECASE)
        end_match = re.search(r'End:\s*([\d:.]+)', block, re.IGNORECASE)
        ad = {}
        if text_match:
            ad_text = text_match.group(1).strip()
            ad['text'] = ad_text
        if start_match:
            ad['start'] = start_match.group(1).strip()
        if end_match:
            ad['end'] = end_match.group(1).strip()
        if ad:
            ads.append(ad)
    return ads

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
