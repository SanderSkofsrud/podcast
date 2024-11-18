# podcast_processor/ad_detection/rate_limiter.py

import time
import threading
import logging
from typing import Dict
from podcast_processor.config import LLM_MODEL_CONFIGS

logger = logging.getLogger(__name__)

class TokenBucket:
    def __init__(self, capacity: int, refill_rate_per_minute: int):
        """
        Initialize a TokenBucket.

        Args:
            capacity (int): Maximum number of tokens in the bucket.
            refill_rate_per_minute (int): Number of tokens added per minute.
        """
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate_per_minute
        self.lock = threading.Lock()
        self.last_refill = time.time()

    def consume(self, tokens: int) -> bool:
        """
        Attempt to consume a number of tokens from the bucket.

        Args:
            tokens (int): Number of tokens to consume.

        Returns:
            bool: True if tokens were consumed, False otherwise.
        """
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_refill
            refill_amount = (elapsed / 60) * self.refill_rate
            if refill_amount >= 1:
                added_tokens = int(refill_amount)
                self.tokens = min(self.capacity, self.tokens + added_tokens)
                self.last_refill = current_time
                logger.debug(f"Refilled {added_tokens} tokens for TokenBucket. Current tokens: {self.tokens}")

            if self.tokens >= tokens:
                self.tokens -= tokens
                logger.debug(f"Consumed {tokens} tokens. Tokens left: {self.tokens}")
                return True
            else:
                logger.debug(f"Not enough tokens. Needed: {tokens}, Available: {self.tokens}")
                return False

    def wait_for_tokens(self, tokens: int):
        """
        Wait until enough tokens are available to consume.

        Args:
            tokens (int): Number of tokens to consume.
        """
        while True:
            if self.consume(tokens):
                return
            sleep_time = 60 / self.refill_rate  # Wait for the next refill
            if sleep_time > 0.05:  # Only log if sleep_time is significant
                logger.info(f"Waiting for {sleep_time:.2f} seconds to refill tokens.")
            time.sleep(sleep_time)

# Initialize TokenBuckets for each model based on config
class RateLimiter:
    def __init__(self, model_configs: Dict[str, Dict]):
        """
        Initialize rate limiters for each model.

        Args:
            model_configs (Dict[str, Dict]): Model configurations from config.py.
        """
        self.buckets = {}
        for model, config in model_configs.items():
            if "max_tokens_per_minute" not in config:
                logger.error(f"Missing 'max_tokens_per_minute' for model '{model}'.")
                continue
            max_tokens = config["max_tokens_per_minute"]
            self.buckets[model] = TokenBucket(
                capacity=max_tokens,
                refill_rate_per_minute=max_tokens
            )
            logger.info(f"Initialized TokenBucket for model '{model}' with capacity {max_tokens} tokens per minute.")

    def acquire(self, model: str, tokens: int):
        """
        Acquire tokens for a specific model.

        Args:
            model (str): The model name.
            tokens (int): Number of tokens to acquire.
        """
        if model not in self.buckets:
            logger.warning(f"No rate limiter found for model '{model}'. Proceeding without rate limiting.")
            return
        logger.info(f"Acquiring {tokens} tokens for model '{model}'...")
        self.buckets[model].wait_for_tokens(tokens)
        logger.info(f"Acquired {tokens} tokens for model '{model}'.")

# Singleton instance
rate_limiter = RateLimiter(LLM_MODEL_CONFIGS)
