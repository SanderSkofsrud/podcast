# classifier.py
from transformers import pipeline
import torch  # Add this import statement
import logging

logger = logging.getLogger(__name__)

classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1",
    device=0 if torch.cuda.is_available() else -1,
    batch_size=8  # Adjust based on your system's capabilities
)

def classify_texts(texts):
    candidate_labels = ["advertisement", "content"]
    results = classifier(texts, candidate_labels=candidate_labels)
    labels = [result['labels'][0] for result in results]
    return labels
