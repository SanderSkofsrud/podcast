# podcast_processor/evaluation/utils.py

import logging

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
