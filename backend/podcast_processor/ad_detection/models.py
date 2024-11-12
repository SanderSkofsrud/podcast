# podcast_processor/ad_detection/models.py

import logging
from typing import Tuple, List

from openai import OpenAI

logger = logging.getLogger(__name__)

def get_llm_models() -> tuple[List, OpenAI]:
    """
    Retrieve the list of LLM models configured for ad detection.
    """
    from podcast_processor.config import LLM_MODELS
    logger.info(f"Using LLM models: {LLM_MODELS}")
    client = OpenAI()
    return LLM_MODELS, client
