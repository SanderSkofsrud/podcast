from llm_evaluation.utils import load_evaluation_data, get_next_run_dir
from llm_evaluation.evaluation import evaluate_model
from llm_evaluation.plotting import plot_metrics
from llm_evaluation.config import LLAMA_MODELS, GPT_MODELS, PLOTS_DIR
import concurrent.futures
import logging

logger = logging.getLogger(__name__)

def main():
    """
    Main function to perform the evaluation.
    """
    # Set up run directory
    run_dir = get_next_run_dir(PLOTS_DIR, prefix="run_llm")

    # Load data
    data = load_evaluation_data()

    results = {}
    all_models = LLAMA_MODELS + GPT_MODELS

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_model = {executor.submit(evaluate_model, model_name, data): model_name for model_name in all_models}

        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                metrics = future.result()
                if metrics:
                    results[model_name] = metrics
            except Exception as e:
                logger.exception(f"Error evaluating model {model_name}: {e}")

    # Plot results
    plot_metrics(results, run_dir)

    logger.info("Evaluation completed.")

if __name__ == "__main__":
    main()
