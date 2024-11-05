# classifier.py

import logging
from config.settings import get_openai_name, client

logger = logging.getLogger(__name__)



def classify_texts(texts):
    labels = []
    for text in texts:
        messages = [
            {"role": "system", "content": "You are an assistant that classifies text as 'advertisement' or 'content'."},
            {"role": "user", "content": f"Classify the following text as 'advertisement' or 'content':\n\n\"{text}\""}
        ]
        try:
            response = client.chat.completions.create(
                model=get_openai_name(),
                messages=messages,
            )
            classification = response['choices'][0]['message']['content'].strip().lower()
            if classification not in ['advertisement', 'content']:
                classification = 'content'  # Default to content if classification is unclear
            labels.append(classification)
        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            labels.append('content')  # Default to content in case of error
    return labels


