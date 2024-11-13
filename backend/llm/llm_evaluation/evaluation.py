from llm_evaluation.models import load_llama_model, load_gpt_model  # Ensure load_gpt_model exists
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Tuple, Dict
import time
import openai
from llm_evaluation.config import LLAMA_MODELS, GPT_MODELS
from openai import OpenAI



logger = logging.getLogger(__name__)

import re

def classify_chunk(client, model_name: str, chunk_text: str) -> int:
    """
    Classifies a text chunk as an advertisement or not using the appropriate client.

    Args:
        client: The client (either OpenAI or Llama) to use for classification.
        model_name (str): The name of the model to use.
        chunk_text (str): The text to classify.

    Returns:
        int: 1 if advertisement, 0 if not.
    """
    for attempt in range(3):  # Retry up to 3 times
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant that identifies whether a given text is an advertisement or not. "
                            "Respond with '1' if the text is an advertisement and '0' if it is not. "
                            "Provide only the number '1' or '0' in your response, without any additional text."
                        )
                    },
                    {"role": "user", "content": chunk_text}
                ]
            )
            classification = response.choices[0].message.content.strip()
            print(f"Model Raw Output: '{classification}'")

            # Use regular expression to extract '1' or '0' from the response
            match = re.search(r'\b(0|1)\b', classification)
            if match:
                return int(match.group(1))
            else:
                logger.warning(f"Unexpected model output: '{classification}'. Retrying...")

        except Exception as e:
            logger.exception(f"Error during classification attempt {attempt + 1}: {e}")

    # If we exhausted retries, return -1 to indicate failure
    logger.error(f"Failed to classify chunk after 3 attempts. Skipping this chunk.")
    return -1


def evaluate_model(model_name: str, data: List[Tuple[str, int]]) -> Dict[str, float]:
    """
    Evaluates the model using the provided data.

    Args:
        model_name (str): The name of the model to evaluate.
        data (list): List of tuples containing the text chunk and the true label.

    Returns:
        dict: Dict with 'precision', 'recall' and 'f1_score'.
    """
    # Determine if it's a Llama or GPT model and load the appropriate client
    if model_name in LLAMA_MODELS:
        client = load_llama_model(model_name)
    elif model_name in GPT_MODELS:
        client = load_gpt_model(model_name)  # This should load OpenAI's GPT models
    else:
        logger.warning(f"Unknown model '{model_name}'. Skipping.")
        return {}

    if client is None:
        return {}

    true_labels = []
    pred_labels = []
    total_time = 0
    count = 0

    for chunk_text, true_label in data:
        start_time = time.time()
        pred_label = classify_chunk(client, model_name, chunk_text)
        end_time = time.time()
        
        if pred_label == -1:
            continue

        true_labels.append(true_label)
        pred_labels.append(pred_label)
        total_time += (end_time - start_time)
        count += 1

    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    avg_time = total_time / count if count > 0 else 0

    logger.info(f"Model: {model_name} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1:.2f} | Avg Time: {avg_time:.2f}s")

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "average_time": avg_time
    }
