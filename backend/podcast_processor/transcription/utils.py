# podcast_processor/transcription/utils.py

import string
import re
import logging
from typing import List, Dict
from thefuzz import process, fuzz

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """
    Normalize text by removing brackets, punctuation, and standardizing whitespace.
    """
    try:
        # Remove content within brackets
        text = re.sub(r'[\[\(\{<].*?[\]\)\}>]', '', text)

        # Replace ellipses with space
        text = re.sub(r'\.{2,}', ' ', text)

        # Convert to lowercase
        text = text.lower()

        # Replace different apostrophe types
        text = text.replace("’", "'").replace("‘", "'").replace("´", "'")

        # Preserve apostrophes in contractions and possessives
        placeholder = ' ###APOSTROPHE### '
        text = re.sub(r"(?<=\w)'(?=\w)", placeholder, text)

        # Remove all punctuation except apostrophes
        punctuation_to_remove = string.punctuation.replace("'", "")
        translator = str.maketrans('', '', punctuation_to_remove)
        text = text.translate(translator)

        # Restore apostrophes
        text = text.replace(' APOSTROPHE ', "'")
        text = text.replace("’", "'")

        # Replace multiple hyphens or dashes with space
        text = re.sub(r'[-–—]{2,}', ' ', text)

        # Remove double dashes and similar patterns
        text = re.sub(r'--+', ' ', text)
        text = re.sub(r'—+', ' ', text)

        # Standardize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Strip leading and trailing whitespace
        text = text.strip()

        return text
    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
        return ""

def load_ground_truth(transcript_path: str) -> str:
    """
    Load and normalize the ground truth transcription from a text file.
    """
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
            normalized_ground_truth = normalize_text(ground_truth)
            logger.debug(f"Loaded and normalized ground truth from '{transcript_path}'.")
            return normalized_ground_truth
    except Exception as e:
        logger.error(f"Error loading ground truth from '{transcript_path}': {e}")
        return ""

def remove_ads_from_transcription(transcription: str, ads: List[Dict]) -> str:
    """
    Remove ads from transcription text.

    Args:
        transcription (str): The transcription text.
        ads (List[Dict]): List of ads detected, each with 'text', 'start', 'end'.

    Returns:
        processed_transcription (str): Transcription with ads removed.
    """
    transcription_normalized = normalize_text(transcription)
    for ad in ads:
        ad_text = ad.get('text', '')
        if ad_text:
            ad_text_normalized = normalize_text(ad_text)
            # Use fuzzy matching to find the ad text in the transcription
            match = process.extractOne(ad_text_normalized, [transcription_normalized], scorer=fuzz.token_set_ratio)
            if match and match[1] >= 80:
                # Remove the matched ad text from the transcription
                match_text = ad_text
                transcription = transcription.replace(match_text, '')
    # Clean up extra spaces
    transcription = ' '.join(transcription.split())
    return transcription
