# podcast_processor/ad_detection/retry.py

import time
import random
import logging
from functools import wraps
from typing import Callable, Any

import openai
from openai import RateLimitError, APIConnectionError, OpenAIError

from podcast_processor.ad_detection.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)

def retry_request(
        max_retries: int = 5,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        allowed_errors: tuple = (RateLimitError, APIConnectionError)
) -> Callable:
    """
    Decorator to retry a function upon encountering specific OpenAI API errors.
    Integrates rate limiting to prevent exceeding max_tokens.

    Args:
        max_retries (int): Maximum number of retry attempts.
        initial_delay (float): Initial delay between retries in seconds.
        backoff_factor (float): Factor by which the delay increases after each retry.
        max_delay (float): Maximum delay between retries in seconds.
        allowed_errors (tuple): Tuple of exception classes that trigger a retry.

    Returns:
        Callable: Wrapped function with retry logic.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            for attempt in range(1, max_retries + 1):
                try:
                    response = func(*args, **kwargs)
                    return response  # Success
                except allowed_errors as e:
                    if attempt == max_retries:
                        logger.error(f"Max retries reached for function '{func.__name__}'.")
                        raise
                    jitter = random.uniform(0, 0.1 * delay)  # Add jitter to delay
                    sleep_time = min(delay * backoff_factor, max_delay) + jitter
                    logger.warning(
                        f"Attempt {attempt} for function '{func.__name__}' failed with error: {e}. "
                        f"Retrying in {sleep_time:.2f} seconds..."
                    )
                    time.sleep(sleep_time)
                except OpenAIError as e:
                    # For other OpenAI errors, do not retry
                    logger.error(f"OpenAIError encountered: {e}. Not retrying.")
                    raise
                except Exception as e:
                    # For all other exceptions, do not retry
                    logger.error(f"Unexpected error: {e}. Not retrying.")
                    raise
        return wrapper
    return decorator
