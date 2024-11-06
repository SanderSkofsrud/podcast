# classifier.py

import logging
import time
import openai
from config.settings import get_openai_name, client

logger = logging.getLogger(__name__)

def classify_texts(texts):
    labels = []
    messages = [
        {"role": "system", "content": "You are an assistant that classifies texts as 'advertisement' or 'content'."}
    ]

    # Prepare the combined text
    combined_texts = ""
    for idx, text in enumerate(texts):
        combined_texts += f"Text {idx+1}:\n{text}\n\n"

    user_content = (
        "Classify each of the following texts as 'advertisement' or 'content'. "
        "Provide your answers in exactly the following format:\n"
        "Text 1: [classification]\n"
        "Text 2: [classification]\n"
        "...\n"
        "Do not include any additional text.\n\n"
        f"{combined_texts}"
    )

    messages.append({"role": "user", "content": user_content})

    max_retries = 5
    retry_delay = 1  # Start with 1 second

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=get_openai_name(),
                messages=messages,
                temperature=0.0  # For deterministic output
            )
            # Use dot notation to access the content
            content = response.choices[0].message.content.strip()

            # Log the assistant's response for debugging
            logger.debug(f"Assistant's response:\n{content}")

            # Parse the response to extract classifications
            labels = []
            for line in content.split('\n'):
                if ':' in line:
                    idx_and_classification = line.split(':', 1)
                    if len(idx_and_classification) == 2:
                        classification = idx_and_classification[1].strip().lower()
                        if classification not in ['advertisement', 'content']:
                            classification = 'content'  # Default to content if classification is unclear
                        labels.append(classification)
            if len(labels) == len(texts):
                break  # Successful classification
            else:
                logger.error(f"Mismatch in number of classifications. Expected {len(texts)}, got {len(labels)}")
                labels = ['content'] * len(texts)  # Default to content
                break
        except openai.error.RateLimitError as e:
            logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        except Exception as e:
            logger.error(f"Error classifying texts: {e}")
            labels = ['content'] * len(texts)  # Default to content in case of error
            break
    else:
        logger.error("Max retries exceeded for classification.")
        labels = ['content'] * len(texts)  # Default to content after max retries

    return labels
