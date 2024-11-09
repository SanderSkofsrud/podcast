# whisper_evaluation/utils.py

import string
import re
import logging

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """
    Normalize text by removing bracketed content, replacing ellipses,
    converting to lowercase, removing punctuation, and standardizing whitespace.

    Args:
        text (str): The text to be normalized.

    Returns:
        str: The normalized text.
    """
    # Remove bracketed content (e.g., [LAUGHS], [LAUGHTER])
    text = re.sub(r'\[.*?\]', '', text)
    # Replace ellipses with a space
    text = re.sub(r'\.\.\.', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    return text.strip()

def load_ground_truth(transcript_path: str) -> str:
    """
    Load the ground truth transcription from a text file.

    Args:
        transcript_path (str): Path to the ground truth transcription file.

    Returns:
        str: The ground truth transcription text.
    """
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
            logger.debug(f"Loaded ground truth from '{transcript_path}'.")
            return ground_truth
    except Exception as e:
        logger.error(f"Error loading ground truth from '{transcript_path}': {e}")
        return ""
