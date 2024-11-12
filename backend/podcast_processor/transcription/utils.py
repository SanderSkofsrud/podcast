# podcast_processor/transcription/utils.py

import string
import re
import logging

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """
    Normalize text by:
    - Removing bracketed content (e.g., [LAUGHS], (HE'S))
    - Replacing ellipses with a space
    - Converting to lowercase
    - Preserving apostrophes in contractions and possessives
    - Removing excessive punctuation
    - Handling hyphens and dashes
    - Standardizing whitespace and newlines
    - **Ignoring patterns like double dashes and similar artifacts**
    """
    try:
        # 1. Remove content within various types of brackets: [], (), {}, <>
        text = re.sub(r'[\[\(\{<].*?[\]\)\}>]', '', text)

        # 2. Replace ellipses (two or more periods) with a single space
        text = re.sub(r'\.{2,}', ' ', text)

        # 3. Convert text to lowercase
        text = text.lower()

        # 4. Replace different apostrophe types with standard apostrophe
        text = text.replace("’", "'").replace("‘", "'").replace("´", "'")

        # 5. Preserve apostrophes in contractions and possessives by temporarily replacing them
        placeholder = ' ###APOSTROPHE### '
        text = re.sub(r"(?<=\w)'(?=\w)", placeholder, text)

        # 6. Remove all punctuation except apostrophes
        punctuation_to_remove = string.punctuation.replace("'", "")
        translator = str.maketrans('', '', punctuation_to_remove)
        text = text.translate(translator)

        # 7. Restore apostrophes
        text = text.replace(' APOSTROPHE ', "'")
        text = text.replace("’", "'")

        # 8. Replace multiple hyphens or dashes with a single space
        text = re.sub(r'[-–—]{2,}', ' ', text)

        # 9. Remove double dashes and similar patterns**
        # This step removes any remaining double dashes or similar artifacts that may not have been caught
        # by the previous hyphen/dash handling step.
        text = re.sub(r'--+', ' ', text)  # Removes sequences like --, ---, etc.
        text = re.sub(r'—+', ' ', text)   # Removes em-dashes if any remain

        # 10. Replace any remaining non-standard whitespace characters with a single space
        text = re.sub(r'\s+', ' ', text)

        # 11. Strip leading and trailing whitespace
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
