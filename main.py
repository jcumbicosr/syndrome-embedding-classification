import argbind
import logging
from pathlib import Path

from src.processing import flatten_data, plot_syndrome_distribution
from src.visualizer import visualize_embeddings
from src.classifier import evaluate_knn
from src.metrics import (
    evaluate_metrics,
    plot_best_k,
    plot_comparative_roc
)

# Set up logging to track pipeline progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MainPipeline")

@argbind.bind(without_prefix=True)
def run_app(
    data_path: str = 'data/mini_gm_public_v0.1.p',
    output_dir: str = 'results',
    perperplexity: int = 30,
    n_splits: int = 10,
    random_state: int = 2,
    show_plots: int = 0,
) -> None:
    """
    Main function to run the syndrome embedding classification pipeline.
    Args:
        data_path: Path to the input data file (pickle format).
        output_dir: Directory to save all outputs (figures and tables).
        perperplexity: Perplexity parameter for t-SNE visualization.
        n_splits: Number of splits for cross-validation.
        random_state: Random state for reproducibility.
        show_plots: Whether to display plots interactively (0 = no, 1 = yes).
    """
    output_dir = Path(output_dir)
    output_fig = output_dir / "figures"
    output_tab = output_dir / "tables"


    logger.info("=== Starting Syndrome Embedding Classification Pipeline ===")


    logger.info("--- Step 1: Data Processing ---")
    df = flatten_data(data_path)
    logging.info(f"Total images: {len(df)}")


    logger.info("--- Step 2: Data Visualization ---")
    plot_syndrome_distribution(df, output_dir=output_fig, show=bool(show_plots))
    visualize_embeddings(df, output_path=output_fig, perplexity=perperplexity, show=bool(show_plots))


    logger.info("--- Step 3: KNN Classification (Cosine Distance)---")
    results_cosine = evaluate_knn(df, distance_metric='cosine', n_splits=n_splits, random_state=random_state)
    cosine_summary = evaluate_metrics(results_cosine, topk_values=[1, 3, 5], tag="cosine", output_dir=output_tab)
    best_k_cosine_dict = plot_best_k(cosine_summary, tag="cosine", output_dir=output_fig, show=bool(show_plots))


    logger.info("--- Step 4: KNN Classification (Euclidean Distance)---")
    results_euclidean = evaluate_knn(df, distance_metric='euclidean', n_splits=n_splits, random_state=random_state)
    euclidean_summary = evaluate_metrics(results_euclidean, topk_values=[1, 3, 5], tag="euclidean", output_dir=output_tab)
    best_k_euclidean_dict = plot_best_k(euclidean_summary, tag="euclidean", output_dir=output_fig, show=bool(show_plots))

    logger.info("--- Step 5: Comparative ROC Analysis ---")
    plot_comparative_roc(results_euclidean=results_euclidean, 
                         results_cosine=results_cosine, 
                         best_k_euclidean=best_k_euclidean_dict['auc_score'], 
                         best_k_cosine=best_k_cosine_dict['auc_score'],
                         output_dir=output_fig, 
                         show=bool(show_plots))
    
    logger.info("=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        run_app()
