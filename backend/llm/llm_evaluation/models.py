import logging
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)


def load_llama_model(model_name: str) -> OpenAI:
    """
    Load the specified Llama model.

    Args:
        model_name (str): The name of the Llama model to load.

    Returns:
        OpenAI: Initialized Llama API client.
    """
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        logger.error("LLAMA_API_KEY not set.")
        raise EnvironmentError("Please set the LLAMA_API_KEY environment variable.")

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.llama-api.com"
        )
        logger.info(f"LLama API model initilized '{model_name}'.")
        return client
    except Exception as e:
        logger.exception(f"Could not init model: {e}")
        return None
    
def load_gpt_model(model_name: str) -> OpenAI:
    """
    Load the specified OpenAI GPT model.

    Args:
        model_name (str): The name of the GPT model to load.

    Returns:
        OpenAI: Initialized GPT client.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set.")
        raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

    try:
        client = OpenAI(api_key=api_key)
        logger.info(f"OpenAI GPT model initialized '{model_name}'.")
        return client
    except Exception as e:
        logger.exception(f"Could not initialize GPT model: {e}")
        return None
    
    
