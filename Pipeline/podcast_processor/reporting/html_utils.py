# podcast_processor/reporting/html_utils.py

import difflib
import re
import logging
from podcast_processor.transcription.utils import normalize_text

logger = logging.getLogger(__name__)

def get_diff_html(reference: str, hypothesis: str) -> str:
    """
    Generate an HTML representation highlighting differences between normalized reference and hypothesis.

    Args:
        reference (str): The ground truth transcription.
        hypothesis (str): The transcribed text to evaluate.

    Returns:
        str: HTML string with differences highlighted.
    """
    try:
        # Normalize both texts
        normalized_ref = normalize_text(reference)
        normalized_hyp = normalize_text(hypothesis)

        # Tokenize the normalized texts into words
        ref_tokens = normalized_ref.split()
        hyp_tokens = normalized_hyp.split()

        # Initialize the HTML output
        html_output = []
        html_output.append("<p>")

        # Use difflib to get the differences
        matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                html_output.append(' '.join(ref_tokens[i1:i2]))
            elif tag == 'replace':
                html_output.append('<span class="replace">' + ' '.join(hyp_tokens[j1:j2]) + '</span>')
            elif tag == 'insert':
                html_output.append('<span class="insert">' + ' '.join(hyp_tokens[j1:j2]) + '</span>')
            elif tag == 'delete':
                html_output.append('<span class="delete">' + ' '.join(ref_tokens[i1:i2]) + '</span>')
            html_output.append(' ')  # Add space between words

        html_output.append("</p>")

        return ''.join(html_output)
    except Exception as e:
        logger.error(f"Error generating diff HTML: {e}")
        return ""
