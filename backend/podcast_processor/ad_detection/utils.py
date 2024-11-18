# podcast_processor/ad_detection/utils.py

import os
import re
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def parse_ads_response(response_text: str) -> List[Dict]:
    """
    Parse the LLM response to extract ads as a list of dictionaries.
    Implements retries if JSON is incomplete.

    Args:
        response_text (str): The raw response from the LLM.

    Returns:
        List[Dict]: A list of ads with 'text', 'start', and 'end'.
    """
    try:
        ads = json.loads(response_text)
        if isinstance(ads, list):
            # Normalize 'ad_text' to 'text' and 'start_time'/'end_time' to 'start'/'end'
            for ad in ads:
                if 'ad_text' in ad and 'text' not in ad:
                    ad['text'] = ad.pop('ad_text')
                if 'start_time' in ad and 'start' not in ad:
                    ad['start'] = ad.pop('start_time')
                if 'end_time' in ad and 'end' not in ad:
                    ad['end'] = ad.pop('end_time')
            return ads
        else:
            logger.error("Parsed JSON is not a list.")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed: {e}. Attempting to extract JSON array manually.")
        # Attempt to extract JSON array from the response using regex
        try:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                ads = json.loads(json_match.group(0))
                if isinstance(ads, list):
                    # Normalize field names
                    for ad in ads:
                        if 'ad_text' in ad and 'text' not in ad:
                            ad['text'] = ad.pop('ad_text')
                        if 'start_time' in ad and 'start' not in ad:
                            ad['start'] = ad.pop('start_time')
                        if 'end_time' in ad and 'end' not in ad:
                            ad['end'] = ad.pop('end_time')
                    return ads
                else:
                    logger.error("Manually extracted JSON is not a list.")
                    return []
            else:
                logger.error("No JSON array found in the response.")
                return []
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Manual JSON extraction failed: {e}")
            return []
    except Exception as e:
        logger.error(f"Unexpected error during JSON parsing: {e}")
        return []

def save_ad_detections(ads: List[Dict], whisper_model: str, llm_model: str, audio_file: str, ad_detections_dir: str):
    """
    Save the detected ads with their timestamps to a JSON file.

    Args:
        ads (List[Dict]): List of detected ads.
        whisper_model (str): The Whisper model used.
        llm_model (str): The LLM model used.
        audio_file (str): The name of the audio file.
        ad_detections_dir (str): Directory to save ad detections.
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

def deduplicate_ads(ads: List[Dict], overlap_threshold: float = 0.5) -> List[Dict]:
    """
    Deduplicate ads by merging overlapping or identical detections.

    Args:
        ads (List[Dict]): List of detected ads.
        overlap_threshold (float): Minimum overlap ratio to consider ads as duplicates.

    Returns:
        List[Dict]: Deduplicated list of ads.
    """
    deduped_ads = []
    for ad in ads:
        # Validate ad structure
        if 'start' not in ad or 'end' not in ad or 'text' not in ad:
            logger.warning(f"Ad missing required fields: {ad}")
            continue

        ad_start = ad['start']
        ad_end = ad['end']

        # Validate types
        if not isinstance(ad_start, (int, float)) or not isinstance(ad_end, (int, float)):
            logger.warning(f"Invalid types for 'start' or 'end' in ad: {ad}")
            continue

        ad_duration = ad_end - ad_start

        # Skip ads with invalid durations
        if ad_duration <= 0:
            logger.warning(f"Ad has non-positive duration: {ad}")
            continue

        overlap_found = False
        for deduped_ad in deduped_ads:
            deduped_start = deduped_ad['start']
            deduped_end = deduped_ad['end']
            deduped_duration = deduped_end - deduped_start

            # Skip deduped ads with invalid duration
            if deduped_duration <= 0:
                logger.warning(f"Deduped ad has non-positive duration: {deduped_ad}")
                continue

            # Calculate overlap
            overlap_start = max(ad_start, deduped_start)
            overlap_end = min(ad_end, deduped_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            # Calculate overlap ratios safely
            if ad_duration > 0 and deduped_duration > 0:
                overlap_ratio_ad = overlap_duration / ad_duration
                overlap_ratio_deduped = overlap_duration / deduped_duration
            else:
                overlap_ratio_ad = 0
                overlap_ratio_deduped = 0

            if overlap_ratio_ad >= overlap_threshold or overlap_ratio_deduped >= overlap_threshold:
                # Merge ads by averaging timestamps and concatenating texts
                deduped_ad['start'] = min(ad_start, deduped_start)
                deduped_ad['end'] = max(ad_end, deduped_end)
                deduped_ad['text'] = f"{deduped_ad['text']} {ad['text']}"
                # Merge models without duplication
                deduped_ad['models'] = list(set(deduped_ad.get('models', []) + ad.get('models', [])))
                overlap_found = True
                break

        if not overlap_found:
            # Add new ad entry with the current model
            new_ad = {
                'text': ad['text'],
                'start': ad_start,
                'end': ad_end,
                'models': ad.get('models', [])
            }
            deduped_ads.append(new_ad)

    return deduped_ads


def validate_ads(ads: List[Dict]) -> List[Dict]:
    """
    Validate and clean the list of ads.

    Args:
        ads (List[Dict]): List of ads to validate.

    Returns:
        List[Dict]: Cleaned list of valid ads.
    """
    valid_ads = []
    for ad in ads:
        text = ad.get('text')
        start = ad.get('start')
        end = ad.get('end')

        # Validate presence of fields
        if text is None or start is None or end is None:
            logger.warning(f"Ad missing required fields: {ad}")
            continue

        # Validate data types
        try:
            start = float(start)
            end = float(end)
        except (ValueError, TypeError):
            logger.warning(f"Invalid 'start' or 'end' time in ad: {ad}")
            continue

        # Validate logical consistency
        if start >= end:
            logger.warning(f"'start' time is not less than 'end' time in ad: {ad}")
            continue

        # Update the ad dictionary with validated values
        ad['start'] = start
        ad['end'] = end
        valid_ads.append(ad)

    return valid_ads
