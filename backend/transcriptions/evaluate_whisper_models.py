# import os
# import time
# import logging
# import warnings
# import torch
# import whisper
# import editdistance
# import matplotlib.pyplot as plt
# import string
# import re
# from multiprocessing import Pool, cpu_count
#
# # ---------------------------
# # Configuration and Setup
# # ---------------------------
#
# # Configure logging to display time, log level, and message
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[
#         logging.StreamHandler()  # Log to console
#     ]
# )
# logger = logging.getLogger(__name__)
#
# # Suppress specific FutureWarnings related to torch.load
# warnings.filterwarnings(
#     "ignore",
#     category=FutureWarning,
#     message=r"You are using `torch.load` with `weights_only=False`"
# )
#
# # Define Whisper model sizes to evaluate
# WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
#
# # Define paths to data directories
# AUDIO_DIR = "data/audio"               # Directory containing podcast audio files
# TRANSCRIPT_DIR = "data/transcripts"    # Directory containing ground truth transcriptions
#
# # Define supported audio file extensions
# SUPPORTED_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac')
#
# # Configure device for model inference (GPU if available, else CPU)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# logger.info(f"Using device: {device}")
#
# # ---------------------------
# # Function Definitions
# # ---------------------------
#
# def load_model(model_name: str, device: str) -> whisper.Whisper:
#     """
#     Load the specified Whisper model onto the designated device.
#
#     Args:
#         model_name (str): The name of the Whisper model to load.
#         device (str): The device to load the model onto ('cuda' or 'cpu').
#
#     Returns:
#         whisper.Whisper: The loaded Whisper model instance.
#     """
#     logger.info(f"Loading Whisper model: {model_name}")
#     try:
#         model = whisper.load_model(model_name, device=device)
#         logger.info(f"Model '{model_name}' loaded successfully.")
#         return model
#     except Exception as e:
#         logger.error(f"Failed to load model '{model_name}': {e}")
#         return None
#
# def transcribe_audio(model: whisper.Whisper, audio_path: str, model_name: str) -> (str, float):
#     """
#     Transcribe the given audio file using the specified Whisper model.
#
#     Args:
#         model (whisper.Whisper): The loaded Whisper model for transcription.
#         audio_path (str): Path to the audio file to be transcribed.
#         model_name (str): Name of the Whisper model being used.
#
#     Returns:
#         tuple:
#             transcription (str): The transcribed text from the audio.
#             elapsed_time (float): Time taken to transcribe the audio in seconds.
#     """
#     try:
#         logger.info(f"Transcribing audio file: '{audio_path}' using model '{model_name}'.")
#         start_time = time.time()
#         result = model.transcribe(
#             audio_path,
#             verbose=False,                    # Disable verbose logging from Whisper
#             fp16=(device == "cuda"),          # Use 16-bit floats if using GPU
#             condition_on_previous_text=False, # Do not condition on previous text
#             temperature=0.0,                  # Deterministic output
#             without_timestamps=False,         # Include timestamps in the output
#             task='transcribe'                 # Specify task as transcription
#         )
#         end_time = time.time()
#         transcription = result["text"].strip()
#         elapsed_time = end_time - start_time
#         logger.info(f"Transcription completed in {elapsed_time:.2f} seconds.")
#         return transcription, elapsed_time
#     except Exception as e:
#         logger.error(f"Error during transcription of file '{audio_path}': {e}")
#         return "", 0.0
#
# def load_ground_truth(transcript_path: str) -> str:
#     """
#     Load the ground truth transcription from a text file.
#
#     Args:
#         transcript_path (str): Path to the ground truth transcription file.
#
#     Returns:
#         str: The ground truth transcription text.
#     """
#     try:
#         with open(transcript_path, 'r', encoding='utf-8') as f:
#             ground_truth = f.read().strip()
#             logger.debug(f"Loaded ground truth from '{transcript_path}'.")
#             return ground_truth
#     except Exception as e:
#         logger.error(f"Error loading ground truth from '{transcript_path}': {e}")
#         return ""
#
# def normalize_text(text: str) -> str:
#     """
#     Normalize text by removing bracketed content, replacing ellipses,
#     converting to lowercase, removing punctuation, and standardizing whitespace.
#
#     Args:
#         text (str): The text to be normalized.
#
#     Returns:
#         str: The normalized text.
#     """
#     # Remove bracketed content (e.g., [LAUGHS], [LAUGHTER])
#     text = re.sub(r'\[.*?\]', '', text)
#     # Replace ellipses with a space
#     text = re.sub(r'\.\.\.', ' ', text)
#     # Convert to lowercase
#     text = text.lower()
#     # Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     # Replace multiple spaces with a single space
#     text = re.sub(r'\s+', ' ', text)
#     # Strip leading and trailing whitespace
#     return text.strip()
#
# def calculate_wer(reference: str, hypothesis: str) -> float:
#     """
#     Calculate Word Error Rate (WER) between reference and hypothesis transcriptions.
#
#     Args:
#         reference (str): The ground truth transcription.
#         hypothesis (str): The transcribed text to evaluate.
#
#     Returns:
#         float: The calculated WER as a float between 0 and 1.
#     """
#     # Normalize both reference and hypothesis texts
#     r = normalize_text(reference).split()
#     h = normalize_text(hypothesis).split()
#     # Compute edit distance between the two word lists
#     distance = editdistance.eval(r, h)
#     # Calculate WER
#     wer_score = distance / len(r) if len(r) > 0 else 0.0
#     return wer_score
#
# def evaluate_model(args: tuple) -> tuple:
#     """
#     Evaluate a single Whisper model across all audio files.
#
#     This function is designed to be used with multiprocessing.
#
#     Args:
#         args (tuple): A tuple containing:
#             - model_name (str): Name of the Whisper model.
#             - model (whisper.Whisper): Loaded Whisper model instance.
#             - audio_files (list): List of audio file names to transcribe.
#
#     Returns:
#         tuple:
#             model_name (str): Name of the Whisper model evaluated.
#             results (dict): Dictionary containing lists of 'speed' and 'accuracy'.
#     """
#     model_name, model, audio_files = args
#     results = {"speed": [], "accuracy": []}
#
#     for audio_file in audio_files:
#         # Skip unsupported file types
#         if not audio_file.lower().endswith(SUPPORTED_EXTENSIONS):
#             logger.debug(f"Skipping unsupported file type: {audio_file}")
#             continue
#
#         audio_path = os.path.join(AUDIO_DIR, audio_file)
#         transcript_file = os.path.splitext(audio_file)[0] + ".txt"
#         transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_file)
#
#         # Check if the corresponding transcript exists
#         if not os.path.exists(transcript_path):
#             logger.warning(f"Missing ground truth for '{audio_file}'. Skipping.")
#             continue
#
#         # Transcribe the audio file
#         transcription, speed = transcribe_audio(model, audio_path, model_name)
#         if transcription == "":
#             logger.warning(f"Transcription failed for '{audio_file}'. Skipping WER calculation.")
#             continue
#         results["speed"].append(speed)
#
#         # Load ground truth transcription
#         ground_truth = load_ground_truth(transcript_path)
#         if ground_truth == "":
#             logger.warning(f"Ground truth loading failed for '{audio_file}'. Skipping WER calculation.")
#             continue
#
#         # Calculate Word Error Rate
#         error_rate = calculate_wer(ground_truth, transcription)
#         accuracy = 1.0 - error_rate  # Higher accuracy is better
#         results["accuracy"].append(accuracy)
#
#         logger.info(f"Model: {model_name} | File: {audio_file} | Speed: {speed:.2f}s | Accuracy: {accuracy:.2f}")
#
#     return model_name, results
#
# def evaluate_models() -> dict:
#     """
#     Evaluate different Whisper models on the dataset using multiprocessing.
#
#     Returns:
#         dict: A dictionary with model names as keys and dictionaries containing
#               'speed' and 'accuracy' lists as values.
#     """
#     # List all audio files in the AUDIO_DIR with supported extensions
#     audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
#     if not audio_files:
#         logger.error("No audio files found in the specified directory.")
#         return {}
#
#     # Load all Whisper models
#     models = {}
#     for model_name in WHISPER_MODELS:
#         model = load_model(model_name, device)
#         if model is not None:
#             models[model_name] = model
#         else:
#             logger.warning(f"Model '{model_name}' could not be loaded and will be skipped.")
#
#     if not models:
#         logger.error("No models were successfully loaded. Exiting evaluation.")
#         return {}
#
#     # Prepare arguments for multiprocessing (model_name, model_instance, audio_files)
#     args = [(model_name, model, audio_files) for model_name, model in models.items()]
#
#     # Determine the number of parallel processes (use number of CPU cores or number of models, whichever is smaller)
#     pool_size = min(len(WHISPER_MODELS), cpu_count())
#     logger.info(f"Starting evaluation with {pool_size} parallel processes.")
#
#     # Initialize multiprocessing pool and evaluate models in parallel
#     with Pool(processes=pool_size) as pool:
#         evaluation_results = pool.map(evaluate_model, args)
#
#     # Compile results into a single dictionary
#     results = {model: {"speed": [], "accuracy": []} for model in WHISPER_MODELS}
#     for model_name, res in evaluation_results:
#         results[model_name]["speed"].extend(res["speed"])
#         results[model_name]["accuracy"].extend(res["accuracy"])
#
#     return results
#
# def aggregate_results(results: dict) -> dict:
#     """
#     Aggregate the results by computing the average speed and accuracy for each model.
#
#     Args:
#         results (dict): Dictionary with model names as keys and dictionaries containing
#                         'speed' and 'accuracy' lists as values.
#
#     Returns:
#         dict: A dictionary containing lists of 'model', 'avg_speed', and 'avg_accuracy'.
#     """
#     aggregated = {"model": [], "avg_speed": [], "avg_accuracy": []}
#     for model, metrics in results.items():
#         if metrics["speed"] and metrics["accuracy"]:
#             avg_speed = sum(metrics["speed"]) / len(metrics["speed"])
#             avg_accuracy = sum(metrics["accuracy"]) / len(metrics["accuracy"])
#             aggregated["model"].append(model)
#             aggregated["avg_speed"].append(avg_speed)
#             aggregated["avg_accuracy"].append(avg_accuracy)
#             logger.info(f"Model: {model} | Avg Speed: {avg_speed:.2f}s | Avg Accuracy: {avg_accuracy:.2f}")
#         else:
#             logger.warning(f"No complete data for model: {model}.")
#     return aggregated
#
# def plot_results(aggregated: dict):
#     """
#     Plot speed vs. accuracy for each Whisper model and save the plot as an image.
#
#     Args:
#         aggregated (dict): Dictionary containing lists of 'model', 'avg_speed', and 'avg_accuracy'.
#     """
#     plt.figure(figsize=(10, 6))
#     plt.scatter(aggregated["avg_speed"], aggregated["avg_accuracy"], color='blue')
#
#     # Annotate each point with the model name
#     for i, model in enumerate(aggregated["model"]):
#         plt.text(
#             aggregated["avg_speed"][i] + 0.02,  # Slight offset for readability
#             aggregated["avg_accuracy"][i] + 0.002,  # Slight offset for readability
#             model,
#             fontsize=9,
#             verticalalignment='bottom'
#         )
#
#     # Set plot labels and title
#     plt.xlabel("Average Transcription Time (s)")
#     plt.ylabel("Average Accuracy (1 - WER)")
#     plt.title("Whisper Models: Speed vs. Accuracy")
#     plt.grid(True)
#     plt.tight_layout()
#
#     # Save and display the plot
#     plt.savefig("whisper_models_speed_vs_accuracy.png")
#     plt.show()
#     logger.info("Results plotted and saved as 'whisper_models_speed_vs_accuracy.png'.")
#
# def main():
#     """
#     Main function to execute the model evaluation pipeline.
#     """
#     start_time = time.time()
#     logger.info("Starting model evaluation...")
#
#     # Step 1: Evaluate models across all audio files
#     results = evaluate_models()
#
#     if not results:
#         logger.error("No results to aggregate. Exiting.")
#         return
#
#     # Step 2: Aggregate the results to compute average metrics
#     aggregated = aggregate_results(results)
#
#     if not aggregated["model"]:
#         logger.error("No aggregated data available. Exiting.")
#         return
#
#     # Step 3: Plot the aggregated results
#     plot_results(aggregated)
#
#     end_time = time.time()
#     total_elapsed = end_time - start_time
#     logger.info(f"Model evaluation completed in {total_elapsed:.2f} seconds.")
#
# # ---------------------------
# # Entry Point
# # ---------------------------
#
# if __name__ == "__main__":
#     main()
