# podcast_processor/ad_detection/token_utils.py

import tiktoken
import logging

logger = logging.getLogger(__name__)

def num_tokens_from_messages(messages, model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    if model_name.startswith("gpt-4"):
        tokens_per_message = 3  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = 1  # If there's a name field
    elif model_name.startswith("gpt-3.5-turbo"):
        tokens_per_message = 4  # For gpt-3.5-turbo
        tokens_per_name = -1  # Should be no name field
    else:
        # For other models or future models, you may need to adjust these numbers
        raise NotImplementedError(f"Token counting not implemented for model {model_name}.")

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 1  
    return num_tokens
