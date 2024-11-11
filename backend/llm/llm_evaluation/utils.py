import os
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

TRANSCRIPTION_DIR = '../data/transcriptions'
ANNOTATION_DIR = '../data/annotations'

def load_evaluation_data() -> List[Tuple[str, int]]:
    """
    Load the evaluation data from the transcription and annotation files.

    Returns:
        list: List of tuples containing the transcription text and the corresponding label.
    """
    data = []
    transcription_files = [f for f in os.listdir(TRANSCRIPTION_DIR) if f.endswith('.txt')]


    for file in transcription_files:
        transcription_path = os.path.join(TRANSCRIPTION_DIR, file)
        annotation_path = os.path.join(ANNOTATION_DIR, file.replace('.txt', '_labels.txt'))

        if not os.path.exists(annotation_path):
            logger.warning(f"Mangler annotasjonsfil for '{file}'. Skipper.")
            continue

        with open(transcription_path, 'r', encoding='utf-8') as t_file, \
             open(annotation_path, 'r', encoding='utf-8') as a_file:
            chunks = t_file.readlines()
            labels = a_file.readlines()

            if len(chunks) != len(labels):
                logger.warning(f"Number of lines '{file}' and '{annotation_path}' doesnt align.")
                continue

            for chunk, label in zip(chunks, labels):
                data.append((chunk.strip(), int(label.strip())))

    print(f"Total data samples loaded: {len(data)}")
    return data



def get_next_run_dir(base_dir: str, prefix: str = "run") -> str:
    """
    Determines the next available run directory (e.g., run_1, run_2, etc.).

    Args:
        base_dir (str): The base directory where run directories are located.
        prefix (str): The prefix for the run directories. Default is "run".

    Returns:
        str: The path to the next available run directory.
    """
    i = 1
    while True:
        run_dir = os.path.join(base_dir, f"{prefix}_{i}")
        if not os.path.exists(run_dir):
            try:
                os.makedirs(run_dir, exist_ok=True)
                logger.info(f"Created directory: {run_dir}")
                return run_dir
            except Exception as e:
                logger.exception(f"Failed to create directory '{run_dir}': {e}")
                raise
        i += 1

