# podcast_processor/evaluation/utils.py

import logging
import os
import json

logger = logging.getLogger(__name__)

def load_ground_truth(transcript_path: str) -> str:
    """
    Load and normalize the ground truth transcription from a text file.
    """
    from podcast_processor.transcription.utils import normalize_text
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
            normalized_ground_truth = normalize_text(ground_truth)
            logger.debug(f"Loaded and normalized ground truth from '{transcript_path}'.")
            return normalized_ground_truth
    except Exception as e:
        logger.error(f"Error loading ground truth from '{transcript_path}': {e}")
        return ""

def load_ground_truth_ads(ground_truth_ads_dir: str) -> dict:
    ground_truth_ads = {}
    for filename in os.listdir(ground_truth_ads_dir):
        if filename.endswith('.json'):
            audio_file = filename.replace('_ads.json', '.mp3')
            filepath = os.path.join(ground_truth_ads_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    ads = json.load(f)
                    ground_truth_ads[audio_file] = ads
                    # Add logging
                    logger.info(f"Loaded ground truth ads for '{audio_file}': {ads}")
            except Exception as e:
                logger.error(f"Error loading ground truth ads from '{filepath}': {e}")
    return ground_truth_ads

