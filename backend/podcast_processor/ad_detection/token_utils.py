# podcast_processor/ad_detection/token_utils.py

import tiktoken
import logging

logger = logging.getLogger(__name__)

def num_tokens_from_messages(messages, model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    if model_name.startswith("gpt-4"):
        tokens_per_message = 3
        tokens_per_name = 1
    elif model_name.startswith("gpt-3.5-turbo"):
        tokens_per_message = 4
        tokens_per_name = -1
    else:
        raise NotImplementedError(f"Token counting not implemented for model {model_name}.")

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 1  # Assistant's reply is primed
    return num_tokens
